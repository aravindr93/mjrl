from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.model_based_baseline import MBBaselineDoubleV
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.algos.npg_off_policy_model_based import NPGOffPolicyModelBased
import mjrl.envs
import time as timer
import os
from shutil import copyfile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from mjrl.algos.model_accel.nn_dynamics import DynamicsModel
from mjrl.utils.TupleMLP import TupleMLP
from mjrl.samplers.core import sample_paths
from mjrl.utils.process_samples import compute_returns
import numpy as np


SEED = 123

e = GymEnv('mjrl_reacher_7dof-v0')

ts=timer.time()

base_dir = "/home/ben/data/off_policy_model/reacher/"
exp_name = "experiment_debug"
exp_dir = os.path.join(base_dir, exp_name)

date = datetime.now()
date_str = date.strftime("%y-%m-%d-%H:%M:%S")
suffix = date_str

try:
    os.makedirs(exp_dir)
except:
    print('skipping makedirs')

job_name = os.path.join(exp_dir, suffix)

# job_name = 'aravind_exp/reacher_off_pol_40_mcl_0'
cur_file = os.path.realpath(__file__)
os.mkdir(job_name)
copyfile(cur_file, os.path.join(job_name, 'source.py'))

writer = SummaryWriter(job_name)

gamma = 0.95
device = 'cuda'

state_dim = e.observation_dim
action_dim = e.action_dim
time_dim = 3

H = 10
NUM_MODELS = 3
NUM_UPDATE_PATHS = 250
NUM_POLICY_UPDATES = 5
VALUE_HIDDEN = 512
DYNAMICS_HIDDEN = 512
UPDATE_PATHS = 250
NUM_TRAJ = 5
NUM_ITER = 20
N_INIT = 10

VALUE_WEIGHT = 0.0
REWARD_WEIGHT = 0.0
# VALUE_WEIGHT = 1.0
# REWARD_WEIGHT = 1.0

BASELINE_BATCH_SIZE = 128

DEBUG=False

if DEBUG:
    NUM_BELLMAN_FIT_ITERS = 10
    NUM_BELLMAN_ITERS = 2
else:
    NUM_BELLMAN_FIT_ITERS = 200
    NUM_BELLMAN_ITERS = 10

#TODO: list
# - add initial paths
# - logging
#    - ficticious returns
# - job script

models = [DynamicsModel(state_dim=e.observation_dim, act_dim=e.action_dim, device=device,
                hidden_size=(DYNAMICS_HIDDEN, DYNAMICS_HIDDEN), seed=SEED+i) for i in range(NUM_MODELS)]

replay_buffer = TrajectoryReplayBuffer(device=device)

policy = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED, init_log_std=-0.5, device=device)

init_paths = sample_paths(N_INIT, e, policy, eval_mode=False, base_seed=SEED + 1)
compute_returns(init_paths, gamma)
replay_buffer.push_many(init_paths)

value_network = TupleMLP(state_dim + time_dim, 1, (VALUE_HIDDEN, VALUE_HIDDEN)).to(device)
no_update_value_network = TupleMLP(state_dim + time_dim, 1, (VALUE_HIDDEN, VALUE_HIDDEN)).to(device)

baseline = MBBaselineDoubleV(models, value_network, no_update_value_network,
    policy, e.env.get_reward, replay_buffer, H, e.horizon, gamma, 
    num_bellman_fit_iters=NUM_BELLMAN_FIT_ITERS, num_bellman_iters=NUM_BELLMAN_ITERS,
    reward_weight=REWARD_WEIGHT, value_weight=VALUE_WEIGHT, batch_size=BASELINE_BATCH_SIZE)

# TODO: wasn't working. may need to use fewer epochs...
model_epoch_losses = baseline.update_all_models(epochs=2)
mean_ep_losses = [np.mean(el) for el in model_epoch_losses]
print('mean epoch losses', mean_ep_losses)

returns_loss = baseline.fit_returns(init_paths, epochs=30)
print('returns loss', returns_loss)

agent = NPGOffPolicyModelBased(e, policy, baseline, 0.05, H, NUM_UPDATE_PATHS,
    NUM_POLICY_UPDATES, gae_lambda=None)

train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=NUM_ITER,
            gamma=gamma,
            gae_lambda=0.97,
            num_cpu=5,
            # sample_mode='samples',
            # num_samples=10 * 1000,
            sample_mode='trajectories',
            num_traj=NUM_TRAJ,
            save_freq=2,
            evaluation_rollouts=25,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score', 'samples'],
            include_iteration=True,
            summary_writer=writer)
print("time taken = %f" % (timer.time()-ts))
