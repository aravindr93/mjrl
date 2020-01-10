from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils.networks import QPi
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.algos.npg_off_policy import NPGOffPolicy

import mjrl.envs
import gym
import robel
import time as timer
SEED = 500

e = GymEnv('DKittyOrientRandom-v0')
policy = MLP(e.spec, hidden_sizes=(256, 256), seed=SEED, init_log_std=-0.5)

replay_buffer = TrajectoryReplayBuffer()

gamma = 0.985

q_function = QPi(policy, e.observation_dim, e.action_dim, 3, e.horizon, replay_buffer,
                hidden_size=(128, 128), batch_size=64, gamma=gamma,
                num_bellman_iters=20, num_fit_iters=100, use_mu_approx=False, num_value_actions=5)
agent = NPGOffPolicy(e, policy, q_function, normalized_step_size=0.05, num_policy_updates=2,
                num_update_states=100, num_update_actions=20, fit_on_policy=True, fit_off_policy=True)

ts=timer.time()

train_agent(job_name='off_policy_npg_dkitty_exp_1',
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
            include_iteration=True)
print("time taken = %f" % (timer.time()-ts))
