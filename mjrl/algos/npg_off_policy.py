

from mjrl.algos.npg_cg import NPG
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QNetwork

class NPGOffPolicy(NPG):

    def __init__(self, env, policy, q_function, normalized_step_size,
                    num_policy_updates,
                    fit_on_policy,
                    fit_off_policy,
                    ):
        super().__init__(env, policy, q_function, normalized_step_size=normalized_step_size)
        self.num_policy_updates = num_policy_updates
        self.fit_on_policy = fit_on_policy
        self.fit_off_policy = fit_off_policy

    def train_step(self, N,
                    env=None,
                    sample_mode='trajectories',
                    horizon=1e6,
                    gamma=0.995,
                    gae_lambda=0.97,
                    num_cpu='max',
                    env_kwargs=None,
                    iteration=None):
        print('train_step NPGOffPolicy')

        if iteration is None:
            raise Exception('Must include iteration number in train_step')

        # get new samples

        # compute returns, advantages

        # push samples to rb
        self.baseline.buffer.push_many(paths)

        # loop over number of policy updates
        for k in range(self.num_policy_updates):
            # fit the Q function
            self.baseline.update()

            # update the policy
            self.update_policy(paths, self.fit_on_policy and k == 0)

    def update_policy(self, paths, fit_on_policy):
        print('update_policy')
        
        if fit_on_policy:
            observations, actions, weights = self.process_paths(paths)
        else:
            pass # fit from the replay buffer.

    def process_paths(self, paths):
        print('process_paths')




