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

env = ML1.get_train_tasks('pick-place-v1')
e = MetaWorldEnv(env)


policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=10, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='metaworld_test_0',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=40,      # samples = 40*25 = 1000
            save_freq=5,
            evaluation_rollouts=10,
            plot_keys=['stoc_pol_mean', 'running_score'])
print("time taken = %f" % (timer.time()-ts))
