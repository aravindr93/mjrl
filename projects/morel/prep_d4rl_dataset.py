"""
Script to convert D4RL dataset into MJRL format
"""

import os
import numpy as np
import pickle
import argparse
import torch
import mjrl.envs
import gym
import d4rl
import mjrl.samplers.core as sampler
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.tensor_utils import d4rl2paths

# ===============================================================================
# Get command line arguments
# ===============================================================================
parser = argparse.ArgumentParser(description='Convert dataset from d4rl format to paths.')
parser.add_argument('--env_name', type=str, required=True, help='environment ID')
parser.add_argument('--output', type=str, required=True, help='location to store data')
parser.add_argument('--act_repeat', type=int, default=1, help='action repeat, will average actions over the repeat')
parser.add_argument('--include', type=str, default=None, help='a file to include (can contain imports and function definitions)')
parser.add_argument('--header', type=str, required=False, help='header commands to execute (can include imports)')
parser.add_argument('--seed', type=int, default=123, help='random seed for sampling')

args = parser.parse_args()
if args.header: exec(args.header)
SEED = args.seed
e = GymEnv(args.env_name)
act_repeat = args.act_repeat
np.random.seed(SEED)
torch.random.manual_seed(SEED)
e.set_seed(SEED)

if args.include:
    import sys
    splits = args.include.split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]: dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    exec("from "+filename+" import *")
if 'obs_mask' in globals(): e.obs_mask = obs_mask

dataset = e.env.env.get_dataset()
raw_paths = d4rl2paths(dataset)

# print some statistics
returns = np.array([np.sum(p['rewards']) for p in raw_paths])
num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
print("Number of samples collected = %i" % num_samples)
print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
       (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )

# prepare trajectory dataset (scaling, transforms etc.)
paths = []
for p in raw_paths:
    path = dict()
    raw_obs = p['observations']
    raw_act = p['actions']
    raw_rew = p['rewards']
    traj_length = raw_obs.shape[0]
    obs = e.obs_mask * raw_obs[::act_repeat]
    act = np.array([np.mean(raw_act[i * act_repeat : (i+1) * act_repeat], axis=0) for i in range(traj_length // act_repeat)])
    rew = np.array([np.sum(raw_rew[i * act_repeat : (i+1) * act_repeat]) for i in range(traj_length // act_repeat)])
    path['observations'] = obs
    path['actions'] = act
    path['rewards'] = rew
    paths.append(path)

pickle.dump(paths, open(args.output, 'wb'))
