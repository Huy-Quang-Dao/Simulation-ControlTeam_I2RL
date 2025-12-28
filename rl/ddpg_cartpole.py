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

from utils.buffers import ReplayBuffer


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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate_actor: float = 1e-4
    """the learning rate of the actor optimizer"""
    learning_rate_critic: float = 1e-3
    """the learning rate of the critic optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the target network update rate (soft update)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 1000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


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


class Actor(nn.Module):
    """Policy network that outputs continuous actions in [-1, 1]"""
    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.single_observation_space.shape).prod()
        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Single continuous action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output in [-1, 1]
        return x


class Critic(nn.Module):
    """Q-network that takes both state and action as input"""
    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.single_observation_space.shape).prod()
        action_size = 1  # Continuous action dimension
        
        self.fc1 = nn.Linear(obs_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize actor-critic networks
    actor = Actor(envs).to(device)
    actor_target = Actor(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate_actor)

    qf = Critic(envs).to(device)
    qf_target = Critic(envs).to(device)
    qf_target.load_state_dict(qf.state_dict())
    q_optimizer = optim.Adam(qf.parameters(), lr=args.learning_rate_critic)

    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),  # Continuous action space
        device,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            # Random continuous action during warmup
            continuous_actions = np.random.uniform(-1, 1, size=(envs.num_envs, 1))
        else:
            with torch.no_grad():
                continuous_actions = actor(torch.Tensor(obs).to(device)).cpu().numpy()
                # Add exploration noise
                noise = np.random.normal(0, args.exploration_noise, size=continuous_actions.shape)
                continuous_actions = np.clip(continuous_actions + noise, -1, 1)

        # Map continuous action to discrete action (0 if < 0, 1 if >= 0)
        discrete_actions = (continuous_actions >= 0).astype(int).flatten()

        # Execute the game
        next_obs, rewards, terminations, truncations, infos = envs.step(discrete_actions)

        # Record rewards for plotting
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Store continuous action in buffer
        rb.add(obs, real_next_obs, continuous_actions, rewards, terminations, infos)

        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            with torch.no_grad():
                # Target policy smoothing
                next_state_actions = actor_target(data.next_observations)
                noise = (torch.randn_like(next_state_actions) * args.exploration_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )
                next_state_actions = (next_state_actions + noise).clamp(-1, 1)
                
                # Compute target Q-value
                target_q = qf_target(data.next_observations, next_state_actions)
                target_q = data.rewards.flatten() + args.gamma * (1 - data.dones.flatten()) * target_q.view(-1)

            # Update critic
            current_q = qf(data.observations, data.actions).view(-1)
            qf_loss = F.mse_loss(current_q, target_q)

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Delayed policy updates
            if global_step % args.policy_frequency == 0:
                # Update actor
                actor_loss = -qf(data.observations, actor(data.observations)).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf_values", current_q.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Evaluation
    eval_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, capture_video=True, run_name=f"{run_name}-eval")]
    )
    
    num_eval_episodes = 10
    eval_epsilon = 0.01
    
    actor.eval()
    eval_returns = []
    
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.Tensor(eval_obs).to(device)
    episode_count = 0
    
    while episode_count < num_eval_episodes:
        with torch.no_grad():
            if random.random() < eval_epsilon:
                continuous_action = np.random.uniform(-1, 1, size=(1, 1))
            else:
                continuous_action = actor(eval_obs).cpu().numpy()
            
            # Map to discrete action
            discrete_action = (continuous_action >= 0).astype(int).flatten()
        
        eval_obs, _, terminations, truncations, infos = eval_env.step(discrete_action)
        eval_obs = torch.Tensor(eval_obs).to(device)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    eval_return = info['episode']['r']
                    eval_returns.append(eval_return)
                    print(f"Eval Episode {episode_count}/{num_eval_episodes}, Return: {eval_return}")
                    writer.add_scalar("eval/episodic_return", eval_return, episode_count)
    
    print(f"Evaluation completed. Mean return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        torch.save({
            'actor': actor.state_dict(),
            'critic': qf.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    eval_env.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()