import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import cvxpy as cp
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = False  # unused, keep rl-style
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 200
    """total timesteps of the experiments"""

    # MPC specific
    horizon: int = 20
    u_max: float = 20.0

    # Eval
    num_eval_episodes: int = 10


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


# MPC Controller
class CartPoleMPC:
    def __init__(self, horizon=20, dt=0.02, u_max=20.0):
        self.N = horizon
        self.dt = dt
        self.u_max = u_max

        # CartPole parameters (Gym standard values)
        self.g = 9.8
        self.mc = 1.0  # mass of cart
        self.mp = 0.1  # mass of pole
        self.l = 0.5   # half-length of pole
        self.total_mass = self.mc + self.mp

        self.nx = 4
        self.nu = 1

        # Cost matrices - penalize theta and theta_dot heavily
        self.Q = np.diag([1.0, 0.1, 100.0, 10.0])  # [x, x_dot, theta, theta_dot]
        self.R = np.diag([0.01])
        self.Qf = np.diag([1.0, 0.1, 200.0, 20.0])  # Terminal cost higher for stability

        self.Ad, self.Bd = self._linear_dynamics()

    def _linear_dynamics(self):
        """
        Linearized dynamics around upright position (theta = 0)
        State: [x, x_dot, theta, theta_dot]
        
        Correct linearization from CartPole equations:
        x_ddot = (u + mp*l*theta_dot^2*sin(theta) - mp*g*sin(theta)*cos(theta)) / 
                 (mc + mp*sin^2(theta))
        theta_ddot = (g*sin(theta) - cos(theta)*x_ddot) / l
        
        At theta = 0: sin(0) = 0, cos(0) = 1
        """
        g = self.g
        mc = self.mc
        mp = self.mp
        l = self.l
        total_mass = mc + mp

        # Linearized continuous-time dynamics
        # dx/dt = A*x + B*u
        A = np.zeros((4, 4))
        A[0, 1] = 1.0  # x_dot
        A[2, 3] = 1.0  # theta_dot
        
        # x_ddot components
        A[1, 2] = -(mp * g) / mc  # theta affects x_ddot
        
        # theta_ddot components
        A[3, 2] = (total_mass * g) / (l * mc)  # gravity term
        
        B = np.zeros((4, 1))
        B[1, 0] = 1.0 / mc  # force affects x_ddot
        B[3, 0] = -1.0 / (l * mc)  # force affects theta_ddot (negative for correct direction)

        # Discretize using forward Euler
        Ad = np.eye(self.nx) + A * self.dt
        Bd = B * self.dt
        
        return Ad, Bd

    def solve(self, x0):
        """
        Solve MPC optimization problem
        Returns continuous control force
        """
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))

        cost = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            # Stage cost
            cost += cp.quad_form(x[:, k], self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            # Dynamics constraints
            constraints += [
                x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k],
            ]
            
            # Control constraints
            constraints += [
                u[:, k] <= self.u_max,
                u[:, k] >= -self.u_max
            ]

        # Terminal cost
        cost += cp.quad_form(x[:, self.N], self.Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if u.value is None:
                # If solver fails, return 0
                return 0.0
            
            return float(u.value[0, 0])
        except:
            return 0.0


def main():
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__MPC__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Training env (no learning, just control)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, False, run_name)]
    )

    obs, _ = envs.reset(seed=args.seed)
    mpc = CartPoleMPC(args.horizon, u_max=args.u_max)

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        x = obs[0]
        u = mpc.solve(x)

        # Map continuous control to discrete action
        # Action 0 = push left (negative force)
        # Action 1 = push right (positive force)
        action = np.array([1 if u > 0 else 0])
        
        obs, rewards, terminations, truncations, infos = envs.step(action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    ep_l = int(info["episode"]["l"])
                    print(f"step={global_step}, return={ep_r:.2f}, length={ep_l}")
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)

        if global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time + 1e-8))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"Step {global_step}, SPS: {sps}")

    envs.close()

    print("\n" + "="*50)
    print("Starting Evaluation...")
    print("="*50)
    
    eval_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + 100, 0, args.capture_video, f"{run_name}-eval")]
    )

    obs, _ = eval_env.reset()
    eval_returns = []
    eval_lengths = []
    episode_count = 0

    while episode_count < args.num_eval_episodes:
        x = obs[0]
        u = mpc.solve(x)
        action = np.array([1 if u > 0 else 0])

        obs, _, terminations, truncations, infos = eval_env.step(action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    ret = float(info["episode"]["r"])
                    length = int(info["episode"]["l"])
                    eval_returns.append(ret)
                    eval_lengths.append(length)
                    print(f"[EVAL] Episode {episode_count}: return={ret:.2f}, length={length}")
                    writer.add_scalar("eval/episodic_return", ret, episode_count)
                    writer.add_scalar("eval/episodic_length", length, episode_count)

    print("\n" + "="*50)
    print(
        f"Evaluation Summary:\n"
        f"  Mean Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}\n"
        f"  Mean Length: {np.mean(eval_lengths):.2f} ± {np.std(eval_lengths):.2f}\n"
        f"  Max Return: {np.max(eval_returns):.2f}\n"
        f"  Min Return: {np.min(eval_returns):.2f}"
    )
    print("="*50)

    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()