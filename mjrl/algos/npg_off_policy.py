
from mjrl.algos.npg_cg import NPG
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QNetwork
from mjrl.utils.cg_solve import cg_solve

import mjrl.samplers.core as trajectory_sampler
import mjrl.utils.process_samples as process_samples
import numpy as np
import time
# TODO: logging

class NPGOffPolicy(NPG):

    def __init__(self, env, policy, q_function, normalized_step_size,
                    num_policy_updates,
                    num_update_states,
                    num_update_actions,
                    fit_on_policy,
                    fit_off_policy,
                    simple_value_func,
                    num_value_actions=1
                    ):
        """
        params:
            env
            policy
            q_function
            normalized_step_size
            num_policy_updates
            num_update_states
            num_update_actions
            fit_on_policy
            fit_off_policy
            simple_value_func - whether to estimate V(s) simply or correctly:
                If true, we estimate V(s) = Q(s,a), a = argmax a' pi(a'|s)
                If false, we estimate V(s) = 1/num_value_states sum_i Q(s,a_i), a_i ~ pi(.|s)
            num_value_states
        """
        super().__init__(env, policy, q_function, normalized_step_size = normalized_step_size)
        self.num_policy_updates = num_policy_updates
        self.num_update_states = num_update_states
        self.num_update_actions = num_update_actions
        self.fit_on_policy = fit_on_policy
        self.fit_off_policy = fit_off_policy
        self.simple_value_func = simple_value_func
        self.num_value_actions = num_value_actions
        if not self.simple_value_func and self.num_value_actions <= 1:
            # TODO: add string logging and scalar logging
            print('warning doing non simple value function approximation with only one update action')

    def train_step(self, N,
                    env=None,
                    sample_mode='trajectories',
                    horizon=1e6,
                    gamma=0.995,
                    gae_lambda=0.97,
                    num_cpu='max',
                    env_kwargs=None,
                    iteration=None):

        if iteration is None:
            raise Exception('Must include iteration number in train_step')

        # get new samples
        env = self.env.env_id if env is None else env
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                                base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_paths(**input_dict)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                                base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_data_batch(**input_dict)

        # compute returns
        process_samples.compute_returns(paths, gamma)

        # push samples to rb
        self.baseline.buffer.push_many(paths)

        # loop over number of policy updates
        for k in range(self.num_policy_updates):
            # fit the Q function
            self.baseline.update_network()

            # update the policy
            self.update_policy(paths, self.fit_on_policy and k == 0)
        
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        return [mean_return, std_return, min_return, max_return]

    def update_policy(self, paths, fit_on_policy):
        if fit_on_policy:
            observations, actions, weights = self.process_paths(paths)
        else:
            observations, actions, weights = self.process_replay()

        # TODO: normalize weights

        # update policy
        surr_before = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]
        alpha, n_step_size, t_gLL, t_FIM, new_params = self.update_from_states_actions(observations, actions, weights)
        
        surr_after = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

    def update_from_states_actions(self, observations, actions, weights):
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------

        # VPG
        ts = time.time()
        vpg_grad = self.flat_vpg(observations, actions, weights)
        t_gLL += time.time() - ts

        # NPG
        ts = time.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += time.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)

        return alpha, n_step_size, t_gLL, t_FIM, new_params

    def process_paths(self, paths):

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        times = np.concatenate([path["time"] for path in paths])

        Qs = self.baseline.predict(observations, actions, times)
        weights = Qs - self.get_value(observations, times)

        return observations, actions, weights
    
    def process_replay(self):
        samples = self.baseline.buffer.get_sample(self.num_update_states)
        observations = samples['observations'].to('cpu').numpy() # TODO: don't convert to/from numpy
        times = samples['time'].to('cpu').numpy()
        actions = self.policy.get_action_batch(observations) 
        Qs = self.baseline.predict(observations, actions, times)
        weights = Qs - self.get_value(observations, times)
        return observations, actions, weights


    def get_value(self, observations, times):
        assert observations.shape[0] == times.shape[0]
        n = observations.shape[0]
        if self.simple_value_func:
            actions, info = self.policy.get_action_batch(observations, return_info=True)
            mean_action = info['mean']
            Q_mean = self.baseline.predict(observations, mean_action, times)
            return Q_mean
        else:
            # observations = np.tile(observations, (self.num_value_actions, 1))
            # times = np.tile(times, (self.num_value_actions))
            values = np.zeros(n)
            for j in range(self.num_value_actions):
                actions = self.policy.get_action_batch(observations)
                Qs = self.baseline.predict(observations, actions, times).reshape(-1)
                values += Qs
            values /= self.num_value_actions
            return values



