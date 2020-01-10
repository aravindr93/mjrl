
from mjrl.baselines.baselines import ReplayBufferBaseline
from mjrl.utils.TupleMLP import TupleMLP

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import copy

class MBBaselineNaive(ReplayBufferBaseline):

    def __init__(self, dynamics_model, value_network, target_value_network, policy,
            reward_func, replay_buffer, H, T, gamma,
            time_dim=3, model_batch_size=128, num_model_fit_iters=300, model_lr=1e-3,
            bellman_batch_size=64, num_bellman_fit_iters=100, num_bellman_iters=10):
        """
            In this version, we will assume all data in the RB is from the same policy
            This allows us to use s, a, s' in the replay buffer instead of 
            sampling new actions and using our learned dynamics model.
            If the data were from mixed policies, the bellman equations would not be valid.
            Will implement the above next after verifying that this works
        """
        super().__init__(replay_buffer)

        self.dynamics_model = dynamics_model
        self.value_network = value_network
        self.target_value_network = target_value_network
        self.policy = policy
        self.reward_func = reward_func

        self.H = H
        self.T = T
        self.gamma = gamma
        self.time_dim = time_dim
        self.model_batch_size = model_batch_size
        self.num_model_fit_iters = num_model_fit_iters
        self.bellman_batch_size = bellman_batch_size
        self.num_bellman_fit_iters = num_bellman_fit_iters
        self.num_bellman_iters = num_bellman_iters

        # should we pass optimizer in instead?
        self.value_optim = torch.optim.Adam(self.value_network.parameters(), 5e-3)

    def update(self):
        model_loss, valid_loss = self.update_model()
        self.bellman_update()

        return {
            'model_loss': model_loss,
            'model_valid_loss': valid_loss
            }        

    def update_model(self):
        print('update model')

        s = np.concatenate([p['observations'][:-1] for p in self.replay_buffer.data_buffer])
        a = np.concatenate([p['actions'][:-1] for p in self.replay_buffer.data_buffer])
        sp = np.concatenate([p['observations'][1:] for p in self.replay_buffer.data_buffer])

        num_valid = 500
        valid_freq = 1

        shuffle_idx = np.random.permutation(s.shape[0])
        s_train = s[shuffle_idx][num_valid:]
        a_train = a[shuffle_idx][num_valid:]
        s_next_train = sp[shuffle_idx][num_valid:]

        s_valid = s[shuffle_idx][:num_valid]
        a_valid = a[shuffle_idx][:num_valid]
        s_next_valid = sp[shuffle_idx][:num_valid]

        model_loss, valid_loss = self.dynamics_model.fit_holdout(s_train, a_train, s_next_train,
            s_valid, a_valid, s_next_valid, self.model_batch_size, self.num_model_fit_iters, valid_freq=valid_freq)
        # model_loss = self.dynamics_model.fit(s, a, sp, self.model_batch_size, self.num_model_fit_iters)

        return model_loss, valid_loss

    def featurize_time(self, t):
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).float()
        t = t.float()
        t = (t+1.0)/self.T
        t_feat = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
        return t_feat

    def featurize_state_time(self, state, time):
        # normalize time
        t = self.featurize_time(time)
        x = torch.cat([state, t], -1)
        return x

    def featurize_state_action(self, state, action):
        x = torch.cat([state, action], -1)
        return x

    def value_no_model(self, state, time, target_network=False):
        x = self.featurize_state_time(state, time)
        if target_network:
            return self.target_value_network(x)
        else:
            return self.value_network(x)

    def value(self, state, time, target_network=False, stop=False):

        s = state
        value = np.zeros(state.size(0))
        for t_offset in range(self.H):

            # TODO: use mean action?
            # TODO: Aravind: add noise, multiple rollouts averaged
            a = self.policy.get_action_pytorch(s).detach()
            # t = time + t_offset
            
            value += self.gamma ** t_offset * self.reward_func(s.detach().numpy(), a.numpy()).reshape(-1)
            
            # x = self.featurize_state_action(s, a)
            # s = s + self.model_network(x)
            s = self.dynamics_model.network(s, a)

        # x = self.featurize_state_time(s, time + self.H)
        if stop:
            import ipdb; ipdb.set_trace()
        return torch.from_numpy(value).float().view(-1, 1) + self.gamma**self.H * self.value_no_model(s, time+self.H, target_network=target_network).detach()


    def compute_bellman_targets(self, idx, *args, **kwargs):
        # For the states with index in the buffer given by idx, compute the Bellman targets
        # If the index corresponds to a terminal state, then the value of next state is 0.0
        amax = np.argmax(idx)
        if idx[amax] == self.replay_buffer['observations'].shape[0]-1:
            idx[(idx == idx[amax])] -= 1
        next_s = self.replay_buffer['observations'][idx+1].detach()
        terminal = self.replay_buffer['is_terminal'][idx].view(-1, 1).detach()
        r = self.replay_buffer['rewards'][idx].detach()
        t = self.replay_buffer['time'][idx].detach()
        bootstrap = self.value(next_s, t + 1, target_network=True).detach().view(-1, 1)
        target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - terminal)
        return target

    def fit_targets(self):
        print('fit_targets')
        for _ in range(self.num_bellman_fit_iters):
            n = self.replay_buffer['observations'].shape[0]
            idx = np.random.permutation(n)[:self.bellman_batch_size]
            targets = self.compute_bellman_targets(idx).detach()
            s = self.replay_buffer['observations'][idx]
            t = self.replay_buffer['time'][idx]

            x = self.featurize_state_time(s, t)

            preds = self.value_network(x)
            self.value_optim.zero_grad()
            loss = F.mse_loss(preds, targets)
            loss.backward()
            self.value_optim.step()

            if _ % 30 == 0:
                print('loss', loss.item())

    def bellman_update(self):
        for _ in range(self.num_bellman_iters):
            self.fit_targets()
            self.target_value_network.load_state_dict(self.value_network.state_dict())


class MBBaselineDoubleV(ReplayBufferBaseline):

    def __init__(self, dynamics_model, value_network, no_update_value_network, policy,
            reward_func, replay_buffer, H, T, gamma,
            time_dim=3, lr=1e-3,
            batch_size=64, num_bellman_fit_iters=100, num_bellman_iters=10,
            reward_weight=1.0, value_weight=1.0):
        """
            In this version, we will assume all data in the RB is from the same policy
            This allows us to use s, a, s' in the replay buffer instead of 
            sampling new actions and using our learned dynamics model.
            If the data were from mixed policies, the bellman equations would not be valid.
            Will implement the above next after verifying that this works
        """
        super().__init__(replay_buffer)

        self.dynamics_model = dynamics_model
        self.value_network = value_network
        self.no_update_value_network = no_update_value_network
        self.policy = policy
        self.reward_func = reward_func

        self.H = H
        self.T = T
        self.gamma = gamma
        self.time_dim = time_dim
        self.batch_size = batch_size
        self.num_bellman_fit_iters = num_bellman_fit_iters
        self.num_bellman_iters = num_bellman_iters
        
        self.reward_weight = reward_weight
        self.value_weight = value_weight

        # should we pass optimizer in instead?
        self.shared_optim = torch.optim.Adam(list(self.value_network.parameters())
            + list(self.dynamics_model.network.parameters()), lr)
    
    def featurize_time(self, t):
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).float()
        t = t.float()
        t = (t+1.0)/self.T
        t_feat = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
        return t_feat

    def featurize_state_time(self, state, time):
        # normalize time
        t = self.featurize_time(time)
        x = torch.cat([state, t], -1)
        return x

    def update(self):
        print('update')

        for _ in range(self.num_bellman_iters):
            self.fit_custom_loss()
            self.no_update_value_network.load_state_dict(self.value_network.state_dict())

    def fit_custom_loss(self):
        print('fit_custom_loss')
        for i in range(self.num_bellman_fit_iters):
            # get samples
            samples = self.replay_buffer.get_sample_safe(self.batch_size)
            s = samples['observations'].detach()
            a = samples['actions'].detach()
            s_next = samples['next_observations'].detach()
            t = samples['time'].detach()

            s_next_hat = self.dynamics_model.network(s, a)

            a_next = self.policy.get_action_pytorch(s_next) #TODO: Noise?
            # construct losses
            reconstruction_loss = F.mse_loss(s_next_hat, s_next)
            reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next), self.reward_func(s_next, a_next))

            #TODO: try below as well
            # a_next_hat = self.policy.get_action_pytorch(s_next_hat)
            # reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next_hat), self.reward_func(s_next, a_next))

            x = self.featurize_state_time(s, t)
            xp = self.featurize_state_time(s_next_hat, t+1)
            V_hat = self.value_network(x)
            target = self.reward_func(s, a).view(-1, 1) + self.gamma * self.value_network(xp)
            value_loss = F.mse_loss(V_hat, target)

            self.shared_optim.zero_grad()
            total_loss = reconstruction_loss + self.reward_weight * reward_loss + self.value_weight * value_loss
            total_loss.backward()
            self.shared_optim.step()

            if i % 25 == 0:
                print('recon', reconstruction_loss.item(), 'rew', reward_loss.item(), 'value', value_loss.item())

    def value_no_model(self, state, time):
        x = self.featurize_state_time(state, time)
        return self.value_network(x)

    def value(self, state, time, use_average=False, n_rollouts=4, add_tvf=True):
        if use_average:
            s = state
            value = np.zeros(state.size(0))
            for t_offset in range(self.H):

                a = self.policy.get_action_pytorch(s, mean_action=True).detach()
                # t = time + t_offset
                
                value += self.gamma ** t_offset * self.reward_func(s.detach().numpy(), a.numpy()).reshape(-1)

                s = self.dynamics_model.network(s, a)
            if add_tvf:
                return torch.from_numpy(value).float().view(-1, 1) + self.gamma**self.H * self.value_no_model(s, time+self.H).detach()
            else:
                return torch.from_numpy(value).float().view(-1, 1)
        else:
            total_value = torch.zeros(state.size(0), 1)
            for r in range(n_rollouts):
                s = state
                value = np.zeros(state.size(0))
                for t_offset in range(self.H):

                    a = self.policy.get_action_pytorch(s, mean_action=False).detach()
                    
                    value += self.gamma ** t_offset * self.reward_func(s.detach().numpy(), a.numpy()).reshape(-1)

                    s = self.dynamics_model.network(s, a)

                if add_tvf:
                   single_val = torch.from_numpy(value).float().view(-1, 1) + self.gamma ** self.H * self.value_no_model(s, time + self.H).detach()
                else:
                    single_val = torch.from_numpy(value).float().view(-1, 1)
            total_value += single_val
            return total_value / n_rollouts