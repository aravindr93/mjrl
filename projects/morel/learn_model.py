"""
Script to learn MDP model from data for offline policy optimization
"""

from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time as timer
import argparse
import os
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Model accelerated policy optimization.')
parser.add_argument('--output', '-o', type=str, required=True, help='location to store the model pickle file')
parser.add_argument('--config', '-c', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--include', '-i', type=str, required=False, help='package to import')
args = parser.parse_args()
with open(args.config, 'r') as f:
    job_data = eval(f.read())
if args.include: exec("import "+args.include)

assert 'data_file' in job_data.keys()
ENV_NAME = job_data['env_name']
SEED = job_data['seed']
del(job_data['seed'])
if 'act_repeat' not in job_data.keys(): job_data['act_repeat'] = 1

# ===============================================================================
# Construct environment and model
# ===============================================================================
if ENV_NAME.split('_')[0] == 'dmc':
    # import only if necessary (not part of package requirements)
    import dmc2gym
    backend, domain, task = ENV_NAME.split('_')
    e = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
    e = GymEnv(e, act_repeat=job_data['act_repeat'])
else:
    e = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
    e.set_seed(SEED)

models = [WorldModel(state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED+i, 
                    **job_data) for i in range(job_data['num_models'])]

# ===============================================================================
# Model training loop
# ===============================================================================

paths = pickle.load(open(job_data['data_file'], 'rb'))
init_states_buffer = [p['observations'][0] for p in paths]
best_perf = -1e8
ts = timer.time()
s = np.concatenate([p['observations'][:-1] for p in paths])
a = np.concatenate([p['actions'][:-1] for p in paths])
sp = np.concatenate([p['observations'][1:] for p in paths])
r = np.concatenate([p['rewards'][:-1] for p in paths])
rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
for i, model in enumerate(models):
    dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)
    loss_general = model.compute_loss(s, a, sp) # generalization error
    if job_data['learn_reward']:
        reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)

pickle.dump(models, open(args.output, 'wb'))
