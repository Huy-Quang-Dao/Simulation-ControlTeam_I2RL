# Implementation of RL Algorithms


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/vwxyzjn/cleanrl)

CS_CLEANRL is an easy-to-use reinforcement learning (RL) framework.

## Installation

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n myenv python=3.8`.

2. To run experiments locally, give the following a try:
git clone https://github.com/Huy-Quang-Dao/Simulation-ControlTeam_I2RL && cd Simulation-ControlTeam_I2RL

3. pip install -r requirements/requirements.txt

## Training
```bash
# `python rl/ppo.py`
python rl/ppo_cartpole.py --seed 1 --env-id CartPole-v0 --total-timesteps 300000

# open another terminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```


