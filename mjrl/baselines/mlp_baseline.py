from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle

class MLPBaseline:
    def __init__(self, env_spec, obs_dim=None, learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, use_gpu=False):
        self.n = obs_dim if obs_dim is not None else env_spec.observation_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.use_gpu = use_gpu

        self.model = nn.Sequential()
        self.model.add_module('fc_0', nn.Linear(self.n+4, 128))
        self.model.add_module('relu_0', nn.ReLU())
        self.model.add_module('fc_1', nn.Linear(128, 128))
        self.model.add_module('relu_1', nn.ReLU())
        self.model.add_module('fc_2', nn.Linear(128, 1))

        if self.use_gpu:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.MSELoss()

    def _features(self, paths):
        o = np.concatenate([path["observations"] for path in paths])
        o = np.clip(o, -10, 10)/10.0
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        N, n = o.shape
        num_feat = int( n + 4 )            # linear + time till pow 4
        feat_mat =  np.ones((N, num_feat)) # memory allocation

        # linear features
        feat_mat[:,:n] = o

        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l)/1000.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al**(j+1)
            k += l
        return feat_mat


    def fit(self, paths, return_errors=False):

        featmat = self._features(paths)
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        featmat = featmat.astype('float32')
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        if self.use_gpu:
            featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
        else:
            featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

        for ep in range(self.epochs):
            rand_idx = np.random.permutation(num_samples)
            for mb in range(int(num_samples / self.batch_size) - 1):
                if self.use_gpu:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]).cuda()
                else:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size])
                batch_x = featmat_var[data_idx]
                batch_y = returns_var[data_idx]
                self.optimizer.zero_grad()
                yhat = self.model(batch_x)
                loss = self.loss_function(yhat, batch_y)
                loss.backward()
                self.optimizer.step()

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def predict(self, path):
        featmat = self._features([path]).astype('float32')
        if self.use_gpu:
            feat_var = Variable(torch.from_numpy(featmat).float().cuda(), requires_grad=False)
            prediction = self.model(feat_var).cpu().data.numpy().ravel()
        else:
            feat_var = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
            prediction = self.model(feat_var).data.numpy().ravel()
        return prediction
