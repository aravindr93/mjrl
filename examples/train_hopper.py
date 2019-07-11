from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import gym
import time as timer
SEED = 500

e = GymEnv('Hopper-v3')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED, init_log_std=-0.5)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='hopper_exp1',
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='trajectories',
            num_traj=40,
            save_freq=10,
            evaluation_rollouts=5,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score', 'samples'])
print("time taken = %f" % (timer.time()-ts))
