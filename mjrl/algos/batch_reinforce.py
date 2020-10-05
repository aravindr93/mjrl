"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import mjrl.samplers.core as sampler
import mjrl.utils.process_samples as process_samples
from torch.autograd import Variable
from mjrl.utils.logger import DataLog


class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.1,
                 seed=123,
                 desired_kl=None,
                 save_logs=False,
                 device='cpu',
                 *args, **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.desired_kl = desired_kl
        self.device = device
        if save_logs: self.logger = DataLog()

    def pg_surrogate(self, observations, actions, advantages):
        # grad of the surrogate is equal to the REINFORCE gradient
        # need to perform ascent on this objective function
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        mean, LL = self.policy.mean_LL(observations, actions)
        adv_var = adv_var.to(LL.device)
        surr = torch.mean(LL*adv_var)
        return surr

    def kl_old_new(self, observations, old_mean, old_log_std, *args, **kwargs):
        new_mean = self.policy.forward(observations)
        new_log_std = self.policy.log_std
        kl_divergence = self.policy.kl_divergence(new_mean, old_mean, new_log_std,
                                                  old_log_std, *args, **kwargs)
        return kl_divergence.to('cpu').data.numpy().ravel()[0]

    def flat_vpg(self, observations, actions, advantages):
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(pg_surr, self.policy.trainable_params)
        return torch.cat([g.contiguous().view(-1) for g in vpg_grad])

    # ----------------------------------------------------------
    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   ):

        # Clean up input arguments
        env = self.env if env is None else env
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()
        self.policy.to('cpu')
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = sampler.sample_paths(**input_dict)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = sampler.sample_data_batch(**input_dict)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        self.policy.to(self.device)
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_before = pg_surr.to('cpu').data.numpy().ravel()[0]
        old_mean = self.policy.forward(observations).detach().clone()
        old_log_std = self.policy.log_std.detach().clone()

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        print(vpg_grad.device)
        t_gLL += timer.time() - ts

        # Policy update with linesearch
        # ------------------------------
        if self.desired_kl is not None:
            max_ctr = 100
            alpha = self.alpha
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params.clone())
                kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)
                if kl_divergence <= self.desired_kl:
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params.clone())
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_after = pg_surr.to('cpu').data.numpy().ravel()[0]
        kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_divergence)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else \
                        0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score


    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)
