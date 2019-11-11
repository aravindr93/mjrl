from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils.networks import QPi
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.algos.npg_off_policy import NPGOffPolicy
from torch.utils.tensorboard import SummaryWriter

import mjrl.envs
import gym
import time as timer
from datetime import datetime
SEED = 500

date = datetime.now()
suffix = date.strftime("%y-%m-%d-%H:%M:%S")
writer = SummaryWriter('/tmp/hopper/exp_{}'.format(suffix))

e = GymEnv('Hopper-v2')
policy = MLP(e.spec, hidden_sizes=(64, 64), seed=SEED, init_log_std=-0.5)

replay_buffer = TrajectoryReplayBuffer()

gamma = 0.9
q_function = QPi(policy, e.observation_dim, e.action_dim, 3, e.horizon, replay_buffer,
                batch_size=512, gamma=gamma, device='cuda',
                num_bellman_iters=10, num_fit_iters=300, fit_lr=1e-3,
                use_mu_approx=True, num_value_actions=5)

agent = NPGOffPolicy(e, policy, q_function, normalized_step_size=0.05, num_policy_updates=2,
                num_update_states=1000*5, num_update_actions=15, fit_on_policy=True, fit_off_policy=True,
                summary_writer=writer)

ts=timer.time()

train_agent(job_name='off_policy_npg_hopper_exp_3',
            agent=agent,
            seed=SEED,
            niter=200,
            gamma=gamma,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='samples',
            num_samples=5 * 1000,
            # sample_mode='trajectories',
            # num_traj=40,
            save_freq=10,
            evaluation_rollouts=5,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score', 'samples'],
            include_iteration=True,
            summary_writer=writer)
print("time taken = %f" % (timer.time()-ts))
