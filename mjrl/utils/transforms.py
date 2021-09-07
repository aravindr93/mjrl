import torch
import numpy as np

class ToCudaTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        for k in sample.keys():
            sample[k] = torch.from_numpy(sample[k])
        return sample

class ClipAction(object):
    """clip action between -1 and 1"""
    def __init__(self, range=[-1, 1]):
        self.range = range

    def __call__(self, sample):
        sample['action'] = np.clip(sample['action'], self.range[0], self.range[1])
        return sample

class ClipObsAddReducedObs(object):
    """clip action between -1 and 1"""
    def __init__(self, relevant_indices=None):
        self.relevant_indices = relevant_indices

    def __call__(self, sample):
        # TODO: make sure the shape only has 1 axis
        sample['reduced_obs'] = np.array(sample['observation'])
        sample['reduced_obs'] = sample['reduced_obs'][self.relevant_indices]
        return sample