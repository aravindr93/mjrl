from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.algos.npg_off_policy import NPGOffPolicy
from mjrl.utils.train_agent import train_agent
from mjrl.utils.networks import QPi
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_point_mass-v0')
policy = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED)
replay_buffer = TrajectoryReplayBuffer()
q_function = QPi(policy, e.observation_dim, e.action_dim, 3, e.horizon, replay_buffer)
agent = NPGOffPolicy(e, policy, q_function, 0.05, 2, 100, 20, True, True, False, 5)

ts = timer.time()
train_agent(job_name='off_policy_pm',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=40,      # samples = 40*25 = 1000
            save_freq=5,
            evaluation_rollouts=None,
            plot_keys=['stoc_pol_mean', 'running_score'], include_iteration=True)
print("time taken = %f" % (timer.time()-ts))
