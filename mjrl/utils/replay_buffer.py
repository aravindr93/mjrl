import numpy as np
import torch
import mjrl.utils.tensor_utils as tensor_utils
from mjrl.utils.process_samples import compute_returns

class TrajectoryReplayBuffer:
    def __init__(self, max_paths=-1, device='cpu'):
        self.data_buffer = []
        self.sampling_mode = 'uniform'
        self.max_paths = max_paths
        self.buffer = dict()
        self.device = device

    def push_many(self, paths):
        for path in paths:
            self.push(path)

    def push_many_with_returns(self, paths, gamma):
        compute_returns(paths, gamma)
        for path in paths:
            self.push(path)

    def push(self, path):
        if self.max_paths != -1:
            while len(self.data_buffer) >= self.max_paths:
                del(self.data_buffer[0])
        self.data_buffer.append(path)
        self.update_buffer(path)

    def update_buffer(self, path):
        for k in path.keys():
            if type(path[k]) == np.ndarray:
                path_data = torch.from_numpy(path[k]).float().to(self.device)
                if k in self.buffer.keys():
                    self.buffer[k] = torch.cat([self.buffer[k], path_data])
                else:
                    self.buffer[k] = path_data

    def get_sample(self, sample_size=1):
        sample_idx = np.random.choice(self.buffer['observations'].shape[0], sample_size)
        sample = dict()
        for k in self.buffer.keys():
            if type(self.buffer[k]) != dict:
                sample[k] = self.buffer[k][sample_idx]
        next_idx = np.clip(sample_idx + 1, 0, self.buffer['observations'].shape[0]-1)
        sample['next_observations'] = self.buffer['observations'][next_idx]
        return sample
    
    def combine_and_remove_terminal(self, samples):
        
        # combine first into one tensor
        all_samples = {}
        for key in samples[0].keys():
            all_samples[key] = torch.cat([a[key] for a in samples])

        keep_idxs = torch.where(1 - all_samples['is_terminal'])
        
        safe_samples = {}

        for key in all_samples:
            safe_samples[key] = all_samples[key][keep_idxs]

        return safe_samples

    def get_sample_safe(self, sample_size=1):
        samples = self.get_sample(sample_size=sample_size)

        total_samples = (1 - samples['is_terminal']).sum()
        all_samples = [samples]

        while total_samples < sample_size:
            x = sample_size - int(total_samples.item())
            more_samples = self.get_sample(sample_size=x)
            total_samples += (1 - more_samples['is_terminal']).sum()
            all_samples.append(more_samples)
        
        return self.combine_and_remove_terminal(all_samples)



    def get_paths(self, sample_size=1):
        path_idx = np.random.choice(len(self.data_buffer), sample_size)
        paths = [self.data_buffer[idx] for idx in path_idx]
        return paths

    def to(self, device='cpu'):
        self.device = device
        for k in self.buffer.keys():
            self.buffer[k] = self.buffer[k].to(device)
        return self

    def is_cuda(self):
        return not(self.device == 'cpu')

    def __getitem__(self, key):
        return self.buffer[key]

if __name__ == '__main__':
    from mjrl.policies.gaussian_mlp import MLP
    from mjrl.utils.gym_env import GymEnv
    from mjrl.samplers.core import sample_paths
    import time
    import mjrl.envs
    e = GymEnv('mjrl_point_mass-v0')
    policy = MLP(e.spec)
    ts = time.time()
    paths = sample_paths(500, e, policy)
    print("sampling time = %f " % (time.time() - ts))
    buffer = TrajectoryReplayBuffer()
    [buffer.push(path) for path in paths]
    import time
    ts = time.time()
    for _ in range(1000):
        buffer.get_sample(sample_size=512)
    print("retrieve time = %f " % (time.time() - ts))
    import ipdb; ipdb.set_trace()