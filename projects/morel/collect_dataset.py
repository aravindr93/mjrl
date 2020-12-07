"""
Script to collect offline dataset from a logging policy
"""

import os
import numpy as np
import pickle
import argparse
import torch
import mjrl.envs
import gym
import mjrl.samplers.core as sampler
from mjrl.utils.gym_env import GymEnv

# ===============================================================================
# Get command line arguments
# ===============================================================================
parser = argparse.ArgumentParser(description='Dataset collection for offline RL.')
parser.add_argument('--env_name', type=str, required=True, help='environment ID')
parser.add_argument('--policy', type=str, required=True, help='path to logging policy pickle file')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--size', type=int, default=int(1e6), help='size of the offline dataset to be collected')
parser.add_argument('--mode', type=str, default='pure', help='exploration mode for dataset collection')
parser.add_argument('--seed', type=int, default=123, help='random seed for sampling')
parser.add_argument('--act_repeat', type=int, default=1, help='action repeat for environment')
parser.add_argument('--include', type=str, default=None, help='a file to include (can contain imports and function definitions)')
parser.add_argument('--header', type=str, required=False, help='header commands to execute (can include imports)')

args = parser.parse_args()
if args.header: exec(args.header)
assert args.mode == 'pure' # Right now, only supporting pure mode of data collection
SEED = args.seed
e = GymEnv(args.env_name, act_repeat=args.act_repeat)
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

policy = pickle.load(open(args.policy, 'rb'))
paths = sampler.sample_data_batch(num_samples=args.size, env=e, policy=policy, eval_mode=False,
                                  base_seed=SEED, num_cpu='max', paths_per_call='max')

# print some statistics
returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print("Number of samples collected = %i" % num_samples)
print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
       (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )

pickle.dump(paths, open(args.output, 'wb'))