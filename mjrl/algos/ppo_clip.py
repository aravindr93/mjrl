import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce import BatchREINFORCE


class PPO(BatchREINFORCE):
    def __init__(self, env, policy, baseline,
                 clip_coef = 0.2,
                 epochs = 10,
                 mb_size = 64,
                 learn_rate = 3e-4,
                 seed = 123,
                 save_logs = False,
                 device = 'cpu',
                 *args, **kwargs,
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.learn_rate = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.clip_coef = clip_coef
        self.epochs = epochs
        self.mb_size = mb_size
        self.device = device
        self.running_score = None
        if save_logs: self.logger = DataLog()

        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=learn_rate)

    def PPO_surrogate(self, observations, actions, advantages, old_LL):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        mean, LL = self.policy.mean_LL(observations, actions)
        LR = torch.exp(LL - old_LL.detach())
        LR_clip = torch.clamp(LR, min=1-self.clip_coef, max=1+self.clip_coef)
        adv_var = adv_var.to(LL.device)
        ppo_surr = torch.mean(torch.min(LR*adv_var,LR_clip*adv_var))
        return ppo_surr

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Optimization algorithm
        # --------------------------
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_before = pg_surr.to('cpu').data.numpy().ravel()[0]
        old_mean = self.policy.forward(observations).detach().clone()
        old_log_std = self.policy.log_std.detach().clone()
        old_LL = self.policy.mean_LL(observations, actions)[1].detach().clone()
        params_before_opt = self.policy.get_param_values()

        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in range(self.epochs):
            for mb in range(int(num_samples / self.mb_size) + 1):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                adv = advantages[rand_idx]
                old_LL_mb = old_LL[rand_idx]
                self.optimizer.zero_grad()
                loss = - self.PPO_surrogate(obs, act, adv, old_LL_mb)
                loss.backward()
                self.optimizer.step()

        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt.clone())
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_after = pg_surr.to('cpu').data.numpy().ravel()[0]
        kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)
        t_opt = timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('t_opt', t_opt)
            self.logger.log_kv('kl_divergence', kl_divergence)
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
