import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32

    # TRPO specific arguments
    delta: float = 0.005
    backtrack_coeff: float = 0.8
    backtrack_iters: int = 10
    cg_iters: int = 10
    damping: float = 0.1
    vf_iters: int = 5

    # runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        n_actions = envs.single_action_space.n

        self.actor = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_probs(self, x):
        return Categorical(logits=self.actor(x))


def conjugate_gradient(Avp_fn, b, cg_iters, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        Avp = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    return x


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    global_step = 0

    for iteration in range(1, args.num_iterations + 1):
        episode_returns = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = torch.tensor(
                np.logical_or(terminations, truncations),
                dtype=torch.float32
            ).to(device)

            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])

        # <<< LOG: episodic return
        if len(episode_returns) > 0:
            mean_return = np.mean(episode_returns)
            writer.add_scalar("charts/episodic_return", mean_return, global_step)
            print(f"Iter {iteration}/{args.num_iterations} | Return: {mean_return:.2f}")

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        with torch.no_grad():
            old_dist = agent.get_action_probs(b_obs)
            old_log_probs = old_dist.log_prob(b_actions)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        new_dist = agent.get_action_probs(b_obs)
        new_log_probs = new_dist.log_prob(b_actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        pg_loss = -(ratio * b_advantages).mean()

        # <<< LOG
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)

        grads = torch.autograd.grad(pg_loss, agent.actor.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()

        def get_kl():
            return torch.distributions.kl_divergence(old_dist, agent.get_action_probs(b_obs)).mean()

        def Fvp(v):
            kl = get_kl()
            grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, agent.actor.parameters())
            return torch.cat([g.view(-1) for g in grads]) + args.damping * v

        stepdir = conjugate_gradient(Fvp, -loss_grad, args.cg_iters)
        shs = 0.5 * torch.dot(stepdir, Fvp(stepdir))
        lm = torch.sqrt(shs / args.delta)
        fullstep = stepdir / lm

        old_params = torch.cat([p.data.view(-1) for p in agent.actor.parameters()])

        def set_flat_params(params):
            idx = 0
            for p in agent.actor.parameters():
                size = p.numel()
                p.data.copy_(params[idx:idx + size].view_as(p))
                idx += size

        for j in range(args.backtrack_iters):
            set_flat_params(old_params + (args.backtrack_coeff ** j) * fullstep)
            with torch.no_grad():
                kl = get_kl()
            if kl <= args.delta:
                break
        else:
            set_flat_params(old_params)

        # value function update (UNCHANGED)
        for _ in range(args.vf_iters):
            v = agent.get_value(b_obs).view(-1)
            v_loss = F.mse_loss(v, b_returns)
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()

        # <<< LOG
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("diagnostics/kl", kl.item(), global_step)

    # Evaluation
    print("\nStarting evaluation...")
    eval_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, capture_video=True, run_name=f"{run_name}-eval")]
    )

    num_eval_episodes = 10
    agent.eval()
    eval_returns = []

    eval_obs, _ = eval_env.reset()
    eval_obs = torch.Tensor(eval_obs).to(device)
    episode_count = 0

    while episode_count < num_eval_episodes:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(eval_obs)

        eval_obs, _, terminations, truncations, infos = eval_env.step(action.cpu().numpy())
        eval_obs = torch.Tensor(eval_obs).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    eval_return = info["episode"]["r"]
                    eval_returns.append(eval_return)
                    print(f"Eval Episode {episode_count}/{num_eval_episodes}, Return: {eval_return}")

    print(f"\nEvaluation completed. Mean return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")

    eval_env.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
