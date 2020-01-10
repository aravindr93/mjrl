from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils.networks import QPi
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.algos.npg_off_policy import NPGOffPolicy

import mjrl.envs
import time as timer
import os
from shutil import copyfile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

SEED = 500

e = GymEnv('mjrl_reacher-v0')

gamma = 0.985
device = 'cuda'



ts=timer.time()

base_dir = "/home/ben/data/off_policy/reacher/"

date = datetime.now()
date_str = date.strftime("%y-%m-%d-%H:%M:%S")
suffix = date_str + "_experiment_0"

try:
    os.makedirs(base_dir)
except:
    print('skipping makedirs')

job_name = os.path.join(base_dir, suffix)

# job_name = 'aravind_exp/reacher_off_pol_40_mcl_0'
cur_file = os.path.realpath(__file__)
os.mkdir(job_name)
copyfile(cur_file, os.path.join(job_name, 'source.py'))

writer = SummaryWriter(job_name)

replay_buffer = TrajectoryReplayBuffer(device=device)
policy = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED, init_log_std=-0.5, device=device)
q_function = QPi(policy, e.observation_dim, e.action_dim, 3, e.horizon, replay_buffer,
                hidden_size=(512, 512), batch_size=4096, gamma=gamma,
                num_bellman_iters=10, num_fit_iters=200, use_mu_approx=True, num_value_actions=5,
                recon_weight=2.0, reward_weight=1.0, device=device)
agent = NPGOffPolicy(e, policy, q_function, normalized_step_size=0.05, num_policy_updates=2,
                num_update_states=2000, num_update_actions=5,
                fit_on_policy=True, fit_off_policy=True, summary_writer=writer)

train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=gamma,
            gae_lambda=0.9,
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
