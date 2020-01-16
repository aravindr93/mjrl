from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.shared_features_baseline import SharedFeaturesBaseline
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.algos.npg_off_policy_shared_features import NPGOffPolicySharedFeatures
import mjrl.envs
import time as timer
import os
from shutil import copyfile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

SEED = 502

e = GymEnv('mjrl_reacher-v0')


ts=timer.time()

base_dir = "/home/ben/data/off_policy_shared/reacher/"
exp_name = "experiment_multiple_actions"
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

gamma = 0.9
device = 'cuda'

HIDDEN_SIZE = 512
hidden_sizes=(HIDDEN_SIZE, HIDDEN_SIZE)
FEAT_SIZE = 64
# HIDDEN_SIZE = 128
# hidden_sizes=(HIDDEN_SIZE,)
# FEAT_SIZE = 128

SHARED_LR = 5e-4
# BOTH_LR = 5e-5 # SDG
BOTH_LR = 1e-3 # ADAM
# BOTH_BELLMAN_LR = 5e-6 # SGD
BOTH_BELLMAN_LR = 1e-3  # ADAM
LINEAR_LR = 1e-3
TARGET_MB_SIZE = 512
BELLMAN_MB = 512    
time_dim = 3
FIT_ITERS = 5000
# BELLMAN_FIT_ITERS = 50
# BELLMAN_ITERS = 5
BELLMAN_FIT_ITERS = 200
BELLMAN_ITERS = 10
MAX_RB = 10

NUM_POLICY_UPDATES = 2

replay_buffers = []

policy = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED, init_log_std=-0.5, device=device)

baseline = SharedFeaturesBaseline(e.observation_dim, e.action_dim, time_dim,
    replay_buffers, policy, e.horizon,
    feature_size=FEAT_SIZE, hidden_sizes=hidden_sizes, shared_lr=SHARED_LR,
    both_lr=BOTH_LR, both_bellman_lr=BOTH_BELLMAN_LR, linear_lr=LINEAR_LR,
    target_minibatch_size=TARGET_MB_SIZE, num_fit_iters=FIT_ITERS, gamma=gamma,
    device=device, bellman_minibatch_size=BELLMAN_MB, num_bellman_fit_iters=BELLMAN_FIT_ITERS,
    num_bellman_iters=BELLMAN_ITERS, max_replay_buffers=MAX_RB)

agent = NPGOffPolicySharedFeatures(e, policy, baseline, normalized_step_size=0.05, num_policy_updates=NUM_POLICY_UPDATES,
                num_update_states=1000, num_update_actions=4,
                fit_on_policy=True, fit_off_policy=True, summary_writer=writer)

train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=gamma,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='samples',
            num_samples=10 * 1000,
            # sample_mode='trajectories',
            # num_traj=40,
            save_freq=10,
            evaluation_rollouts=5,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score', 'samples'],
            include_iteration=True,
            summary_writer=writer)
print("time taken = %f" % (timer.time()-ts))
