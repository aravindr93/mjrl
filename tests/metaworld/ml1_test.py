from mjrl.utils.gym_env import MetaWorldEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer

from metaworld.benchmarks import ML1

SEED = 500

env = ML1.get_train_tasks('bin-picking-v1')
e = MetaWorldEnv(env)


policy = MLP(e.spec, hidden_sizes=(256,256), seed=SEED, init_log_std=0.0)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=10, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.01, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='metaworld_bin_picking_4',
            agent=agent,
            seed=SEED,
            niter=200,
            gamma=0.99,
            gae_lambda=0.99,
            num_cpu=16,
            sample_mode='trajectories',
            num_traj=500,      # samples = 40*25 = 1000
            save_freq=50,
            evaluation_rollouts=10,
            plot_keys=['stoc_pol_mean', 'running_score'])
print("time taken = %f" % (timer.time()-ts))
