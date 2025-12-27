import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


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
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3   # <<< CHANGED (REINFORCE-appropriate)
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class PolicyNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, x):
        logits = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)


def compute_returns(rewards, gamma):
    returns = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns   # <<< CHANGED (remove normalization)


def main():
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    # Agent
    agent = PolicyNetwork(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    obs, _ = envs.reset(seed=args.seed)

    episode_rewards = []
    episode_log_probs = []

    global_step = 0
    start_time = time.time()

    while global_step < args.total_timesteps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        action, log_prob = agent.get_action(obs_tensor)
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        episode_rewards.append(torch.tensor(reward, device=device))
        episode_log_probs.append(log_prob)

        obs = next_obs
        global_step += args.num_envs

        if done.any():
            rewards = torch.cat(episode_rewards)
            log_probs = torch.cat(episode_log_probs)

            returns = compute_returns(rewards, args.gamma)
            loss = -(log_probs * returns).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for info in infos.get("final_info", []):
                if info and "episode" in info:
                    writer.add_scalar(
                        "charts/episodic_return",
                        info["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    episode_return = info["episode"]["r"].item()
                    episode_length = info["episode"]["l"].item()

                    print(
                        f"step={global_step}, return={episode_return:.2f}, length={episode_length}"
                    )

            episode_rewards.clear()
            episode_log_probs.clear()
            obs, _ = envs.reset()

    eval_env = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                seed=args.seed,
                idx=0,
                capture_video=True,
                run_name=f"{run_name}-eval",
            )
        ]
    )

    num_eval_episodes = 10

    agent.eval()
    eval_returns = []

    eval_obs, _ = eval_env.reset(seed=args.seed)
    eval_obs = torch.Tensor(eval_obs).to(device)
    episode_count = 0

    while episode_count < num_eval_episodes:
        with torch.no_grad():
            logits = agent(eval_obs)
            eval_action = torch.argmax(logits, dim=1).cpu().numpy()

        eval_obs, _, terminations, truncations, infos = eval_env.step(eval_action)
        eval_obs = torch.Tensor(eval_obs).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    eval_return = info["episode"]["r"]
                    eval_returns.append(eval_return)
                    print(f"Eval Episode {episode_count}/{num_eval_episodes}, Return: {eval_return}")
                    writer.add_scalar("eval/episodic_return", eval_return, episode_count)

    print("\n" + "=" * 50)
    print(f"Evaluation completed!")
    print(f"Mean return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print("=" * 50 + "\n")

    eval_env.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
