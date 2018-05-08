"""
    Add the environment you wish to train here
"""

import gym
from mjrl.utils.gym_env import GymEnv

def get_environment(env_name=None):
    if env_name is None: print("Need to specify environment name")
    return GymEnv(env_name)
