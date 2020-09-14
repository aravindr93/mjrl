import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import os
import time as timer
from torch.autograd import Variable
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.replay_buffer import ReplayBuffer
from mjrl.algos.model_accel.nn_dynamics import WorldModel
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.algos.model_accel.sampling import policy_rollout
from mjrl.utils.cg_solve import cg_solve

# Import NPG
from mjrl.algos.npg_cg import NPG


class NPGSAC(NPG):
    def __init__(self, 
                 gamma=0.99,
                 replay_buffer_size=1e6,
                 **kwargs):
        super(NPGSAC, self).__init__(**kwargs)

        self._gamma = gamma
        self._actor_batchsize = 2**14
        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        
        return

    def to(self, device):
        # Convert all the networks (except policy network which is clamped to CPU)
        # to the specified device
        try:    
            self.baseline.model.to(device)
        except: 
            pass
        return

    def is_cuda(self):
        # Check if any of the networks are on GPU
        baseline_cuda = next(self.baseline.model.parameters()).is_cuda
        return baseline_cuda

    def store(self, paths):
        s = np.concatenate([p['observations'][:-1] for p in paths])
        a = np.concatenate([p['actions'][:-1] for p in paths])
        sp = np.concatenate([p['observations'][1:] for p in paths])
        r = np.concatenate([p['rewards'][:-1] for p in paths])
        terminated = np.concatenate([[False] * (len(p['observations']) - 2) + [p['terminated']] for p in paths])
                
        self._replay_buffer.store(s=s, a=a, r=r, sp=sp, terminated=terminated)
        
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)

        return

    def train_step(self, num_new_samples):
        ts = timer.time()

        critic_steps = num_new_samples
        self._update_critic(critic_steps)

        self._update_actor(self._actor_batchsize)

        return

    def get_action(self, observation):
        return self.policy.get_action(observation)

    def get_refined_action(self, observation):
        # TODO(Aravind): Implemenet this
        # This function should rollout many trajectories according to the learned
        # dynamics model and the policy, and should refine around the policy by
        # incorporating reward based refinement
        raise NotImplementedError

    def _update_critic(self, steps):
        num_samples = steps * self.baseline.batch_size
        s, a, r, sp, terminated = self._replay_buffer.sample(num_samples)
        tar_vals = self._compute_tar_vals(r, sp, terminated)
        tar_vals = np.expand_dims(tar_vals, axis=-1)

        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(s, a, tar_vals, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(s, a, tar_vals)

        return

    def _update_actor(self, batchsize):
        s, _, _, _, _ = self._replay_buffer.sample(batchsize)
        
        s_torch = torch.from_numpy(s).float()
        a_torch = self.policy.model.forward(s_torch)
        a_torch = a_torch + torch.randn(a_torch.shape) * torch.exp(self.policy.log_std)
        a = a_torch.detach().numpy()

        adv = self._computer_advantages(s=s, a=a)

        self.train_from_batch(s, a, adv)

        return

    def _compute_tar_vals(self, r, sp, terminated):
        sp_torch = torch.from_numpy(sp).float()
        ap_torch = self.policy.model.forward(sp_torch)
        ap_torch = ap_torch + torch.randn(ap_torch.shape) * torch.exp(self.policy.log_std)
        ap = ap_torch.detach().numpy()

        not_terminated = 1.0 - terminated.astype(np.float32)
        next_val = self.baseline.predict(sp, ap)
        new_val = r + self._gamma * not_terminated * next_val

        return new_val

    def _computer_advantages(self, s, a):
        vals = self.baseline.predict(s, a)
        adv = vals
        return adv

    def train_from_batch(self, s, a, adv):
        observations = s
        actions = a
        advantages = adv

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # normalize inputs if necessary
        if self.input_normalization:
            data_in_shift, data_in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            pi_in_shift, pi_in_scale = self.policy.model.in_shift.data.numpy(), self.policy.model.in_scale.data.numpy()
            pi_out_shift, pi_out_scale = self.policy.model.out_shift.data.numpy(), self.policy.model.out_scale.data.numpy()
            pi_in_shift = self.input_normalization * pi_in_shift + (1-self.input_normalization) * data_in_shift
            pi_in_scale = self.input_normalization * pi_in_scale + (1-self.input_normalization) * data_in_scale
            self.policy.model.set_transformations(pi_in_shift, pi_in_scale, pi_out_shift, pi_out_scale)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

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
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return