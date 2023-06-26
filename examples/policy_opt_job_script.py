"""
This is a job script for running policy gradient algorithms on gym tasks.
Separate job scripts are provided to run few other algorithms
- For DAPG see here: https://github.com/aravindr93/hand_dapg/tree/master/dapg/examples
- For model-based NPG see here: https://github.com/aravindr93/mjrl/tree/master/mjrl/algos/model_accel
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
from mjrl.utils.logger import DataLog
import os
import json
import gym
import mjrl.envs
import time as timer
import pickle
import argparse

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Natural policy gradient from mjrl on mujoco environments')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
args = parser.parse_args()
JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'NVPG', 'VPG', 'PPO']])
assert 'sample_mode' in job_data.keys()
job_data['alg_hyper_params'] = dict() if 'alg_hyper_params' not in job_data.keys() else job_data['alg_hyper_params']

EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

if job_data['sample_mode'] == 'trajectories':
    assert 'rl_num_traj' in job_data.keys()
    job_data['rl_num_samples'] = 0 # will be ignored
elif job_data['sample_mode'] == 'samples':
    assert 'rl_num_samples' in job_data.keys()
    job_data['rl_num_traj'] = 0    # will be ignored
else:
    print("Unknown sampling mode. Choose either trajectories or samples")
    exit()

# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'])
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'], init_log_std=job_data['init_log_std'])
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'], hidden_sizes=job_data['vf_hidden_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

# Construct the algorithm
if job_data['algorithm'] == 'NPG':
    # Other hyperparameters (like number of CG steps) can be specified in config for pass through
    # or default hyperparameters will be used
    agent = NPG(e, policy, baseline, normalized_step_size=job_data['rl_step_size'],
                seed=job_data['seed'], save_logs=True, **job_data['alg_hyper_params'])

elif job_data['algorithm'] == 'VPG':
    agent = BatchREINFORCE(e, policy, baseline, learn_rate=job_data['rl_step_size'],
                           seed=job_data['seed'], save_logs=True, **job_data['alg_hyper_params'])

elif job_data['algorithm'] == 'NVPG':
    agent = BatchREINFORCE(e, policy, baseline, desired_kl=job_data['rl_step_size'],
                           seed=job_data['seed'], save_logs=True, **job_data['alg_hyper_params'])

elif job_data['algorithm'] == 'PPO':
    # There are many hyperparameters for PPO. They can be specified in config for pass through
    # or defaults in the PPO algorithm will be used
    agent = PPO(e, policy, baseline, save_logs=True, **job_data['alg_hyper_params'])


# Update logger if WandB in Config
if 'wandb_params' in job_data.keys() and job_data['wandb_params']['use_wandb']==True:
    if 'wandb_logdir' in job_data['wandb_params']:
        job_data['wandb_params']['wandb_logdir'] = os.path.join(JOB_DIR, job_data['wandb_params']['wandb_logdir'])
    else:
        job_data['wandb_params']['wandb_logdir'] = JOB_DIR
    agent.logger = DataLog(**job_data['wandb_params'], wandb_config=job_data)


print("========================================")
print("Starting policy learning")
print("========================================")

ts = timer.time()
train_agent(job_name=JOB_DIR,
            agent=agent,
            seed=job_data['seed'],
            niter=job_data['rl_num_iter'],
            gamma=job_data['rl_gamma'],
            gae_lambda=job_data['rl_gae'],
            num_cpu=job_data['num_cpu'],
            sample_mode=job_data['sample_mode'],
            num_traj=job_data['rl_num_traj'],
            num_samples=job_data['rl_num_samples'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))
