import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm
from mjrl.algos.behavior_cloning import BC as BaseBC


class BC(BaseBC):
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

        in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr) if optimizer is None else optimizer

        # loss criterion is MSE for maximum likelihood estimation
        self.loss_function = torch.nn.MSELoss()

    def loss(self, data, idx=None):
        idx = range(data['observations'].shape[0]) if idx is None else idx
        obs = data['observations'][idx]
        obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        act_expert = data['expert_actions'][idx]
        act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
        act_pi = self.policy.model(obs)
        return self.loss_function(act_pi, act_expert.detach())