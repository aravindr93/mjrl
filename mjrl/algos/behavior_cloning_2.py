import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm

class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()

        # get transformations
        observations = np.concatenate([path["observations"] for path in expert_paths])
        actions = np.concatenate([path["actions"] for path in expert_paths])
        in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
        out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)

        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr) if optimizer is None else optimizer

        # loss criterion is MSE for maximum likelihood estimation
        self.loss_function = torch.nn.MSELoss()

    def loss(self, obs, act):
        obs_var = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        act_var = Variable(torch.from_numpy(act).float(), requires_grad=False)
        act_hat = self.policy.model(obs_var)
        return self.loss_function(act_hat, act_var.detach())

    def train(self):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        actions = np.concatenate([path["actions"] for path in self.expert_paths])

        params_before_opt = self.policy.get_param_values()
        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in tqdm(range(self.epochs)):
            self.logger.log_kv('epoch', ep)
            loss_val = self.loss(observations, actions).data.numpy().ravel()[0]
            self.logger.log_kv('loss', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                self.optimizer.zero_grad()
                loss = self.loss(obs, act)
                loss.backward()
                self.optimizer.step()
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        self.logger.log_kv('epoch', self.epochs)
        loss_val = self.loss(observations, actions).data.numpy().ravel()[0]
        self.logger.log_kv('loss', loss_val)
        self.logger.log_kv('time', (timer.time()-ts))
