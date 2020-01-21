from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import gym
import time as timer

import os
from shutil import copyfile

SEED = 500

e = GymEnv('Swimmer-v2')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED, init_log_std=-0.5)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts=timer.time()

job_name='/home/ben/data/npg/swimmer/exp_baseline1'

cur_file = os.path.realpath(__file__)
os.mkdir(job_name)
copyfile(cur_file, os.path.join(job_name, 'source.py'))


train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.99,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='samples',
            num_samples=20 * 1000,
            save_freq=10,
            evaluation_rollouts=5,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score', 'samples'])
print("time taken = %f" % (timer.time()-ts))
