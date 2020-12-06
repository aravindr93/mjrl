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
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

# Import NPG
from mjrl.algos.npg_cg import NPG

class TRPO(NPG):
    def __init__(self, env, policy, baseline,
                 kl_divergence=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 normalized_step_size=0.01,
                 device='cpu',
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_divergence: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_divergence = kl_divergence if kl_divergence is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.device = device
        if save_logs: self.logger = DataLog()

    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_before = pg_surr.to('cpu').data.numpy().ravel()[0]
        old_mean = self.policy.forward(observations).detach().clone()
        old_log_std = self.policy.log_std.detach().clone()

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'],
                                  device=vpg_grad.device)
        npg_grad = cg_solve(f_Ax=hvp, b=vpg_grad, x_0=vpg_grad.clone(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        n_step_size = 2.0*self.kl_divergence
        inner_prod = vpg_grad.dot(npg_grad)
        alpha = torch.sqrt(torch.abs(n_step_size / (inner_prod + 1e-10)))
        alpha = alpha.to('cpu').data.numpy().ravel()[0]

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        for k in range(100):
            new_params = curr_params + alpha * npg_grad
            self.policy.set_param_values(new_params.clone())
            surr_after = pg_surr.to('cpu').data.numpy().ravel()[0]
            kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)
            if kl_divergence < self.kl_divergence:
                break
            else:
                alpha = 0.9*alpha # backtrack
                print("Step size too high. Backtracking. | kl = %f | surr diff = %f" % \
                      (kl_divergence, surr_after-surr_before) )
            if k == 99:
                alpha = 0.0

        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params.clone())
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_after = pg_surr.to('cpu').data.numpy().ravel()[0]
        kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
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