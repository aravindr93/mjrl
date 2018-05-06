from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_point_mass-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = QuadraticBaseline(e.spec)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='point_mass_exp1',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.95,       # approx 1-(1/horizon)
            gae_lambda=0.97,  # from paper
            num_cpu=4,
            sample_mode='trajectories',
            num_traj=40,      # samples = 40*25 = 1000
            save_freq=5,
            evaluation_rollouts=10)
print("time taken = %f" % (timer.time()-ts))
