import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.optimize_model import fit_data

import pickle

class MLPQBaseline(MLPBaseline):
    def __init__(self, env_spec, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, use_gpu=False, hidden_sizes=(128, 128)):
        super().__init__(env_spec, inp_dim, inp, learn_rate, reg_coef,
                            batch_size, epochs, use_gpu, hidden_sizes)
        return

    def _features(self, paths):
        if self.inp == 'env_features':
            o = np.concatenate([path["env_infos"]["env_features"][0] for path in paths])
        else:
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