
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np

from mjrl.baselines.baselines import Baseline
from mjrl.utils.TupleMLP import TupleMLP

import time

class SharedFeaturesBaseline(Baseline):

    def __init__(self, state_dim, action_dim, time_dim, replay_buffers, policy, T,
                feature_size=64, hidden_sizes=(64, 64), both_lr=1e-3, shared_lr=1e-3, linear_lr=1e-3, both_bellman_lr=1e-3, target_minibatch_size=64,
                num_fit_iters=200, num_bellman_iters=10, num_bellman_fit_iters=200, bellman_minibatch_size=64,
                bias=True, gamma=0.95, total_loss_weight=1.0, device='cpu', max_replay_buffers=-1):
        self.time_dim = time_dim
        self.replay_buffers = replay_buffers
        self.policy = policy
        self.T = T
        self.K = len(replay_buffers)
        self.max_replay_buffers = max_replay_buffers
        self.feature_size = feature_size
        self.bias = bias
        self.shared_lr = shared_lr
        self.both_lr = both_lr
        self.linear_lr = linear_lr
        self.both_bellman_lr = both_bellman_lr

        self.shared_features_network = TupleMLP(state_dim + action_dim + time_dim, self.feature_size, hidden_sizes=hidden_sizes).to(device)
        self.policy_linear = torch.nn.Linear(self.feature_size, 1, bias=bias).to(device)

        self.target_shared_features_network = TupleMLP(state_dim + action_dim + time_dim, self.feature_size, hidden_sizes=hidden_sizes).to(device)
        self.target_policy_linear = torch.nn.Linear(self.feature_size, 1, bias=bias).to(device)

        self.target_minibatch_size = target_minibatch_size
        self.bellman_minibatch_size = bellman_minibatch_size
        self.num_fit_iters = num_fit_iters
        self.num_bellman_iters = num_bellman_iters
        self.num_bellman_fit_iters = num_bellman_fit_iters
        self.gamma = gamma
        self.total_loss_weight = total_loss_weight

        self.device = device

        self.linear_q_weights = [torch.nn.Linear(self.feature_size, 1, bias=bias).to(self.device) for _ in range(self.K)]
        self.all_params = []
        for weight in self.linear_q_weights:
            for param in weight.parameters():
                self.all_params.append(param)
        self.both_optim=optim.Adam(list(self.shared_features_network.parameters()) + self.all_params, both_lr)
        # self.both_optim=optim.SGD(list(self.shared_features_network.parameters()) + self.all_params, both_lr, momentum=0.9)
    
    def featurize_time(self, t):
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).float().to(self.device)
        if self.time_dim == 0:
            return torch.zeros(t.shape[0], 0).to(self.device)
        t = t.float()
        t = (t+1.0)/self.T
        t_feat = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
        return t_feat

    def featurize_state_time(self, state, time):
        t = self.featurize_time(time)
        x = torch.cat([state, t], -1)
        return x

    def featurize_state_action_time(self, state, action, time):
        # normalize time
        t = self.featurize_time(time)

        if type(state) == np.ndarray:
            state = torch.from_numpy(state).to(self.device).float()
        if type(action) == np.ndarray:
            action = torch.from_numpy(action).to(self.device).float()

        x = torch.cat([state, action, t], -1)
        return x
    
    def sample_replay_buffers(self, sample_size, p=None):
        # TODO: this can be done more efficiently
        # sample the policies uniformly, then sample data from those policies uniformly
        # could do a weighted sample letting p = size(rb_i) / sum(size(rb_i))
        # could also prefer newer data
        if p is None:
            policies = np.random.choice(self.K, sample_size)
        else:
            policies = np.random.choice(self.K, sample_size, p=p)

        samples = []
        for policy in policies:
            # sample = self.replay_buffers[policy].get_sample_safe() #TODO: which to use here?
            sample = self.replay_buffers[policy].get_sample()
            samples.append(sample)

        all_samples = {}
        for k in samples[0].keys():
            data = []
            for sample in samples:
                data.append(sample[k])
            all_samples[k] = torch.cat(data).detach()
        
        return all_samples, policies

    def sample_replay_buffers_efficient(self, sample_size, p=None):
        # sample the policies uniformly, then sample data from those policies uniformly
        # could do a weighted sample letting p = size(rb_i) / sum(size(rb_i))
        # could also prefer newer data
        if p is None:
            policies = np.random.choice(self.K, sample_size)
        else:
            policies = np.random.choice(self.K, sample_size, p=p)
        samples = []

        counts = np.zeros(self.K, dtype=int)

        for policy in policies:
            counts[policy] += 1


        for i in range(self.K):
            sample = self.replay_buffers[i].get_sample(counts[i])
            samples.append(sample)

        all_samples = {}
        for k in samples[0].keys():
            data = []
            for sample in samples:
                data.append(sample[k])
            all_samples[k] = torch.cat(data).detach()
                
        return all_samples, counts

    def update_returns(self):
        print('update_returns')
        losses = []

        total_sample_time = 0.0
        total_loss_time = 0.0

        for _ in range(self.num_fit_iters):
            start_time = time.time()
            
            # all_samples, policies = self.sample_replay_buffers(self.target_minibatch_size)
            all_samples, counts = self.sample_replay_buffers_efficient(self.target_minibatch_size)
            total_sample_time += (time.time() - start_time)
            s = all_samples['observations']
            a = all_samples['actions']
            t = all_samples['time']
            returns = all_samples['returns']
            x = self.featurize_state_action_time(s, a, t)
            
            self.both_optim.zero_grad()

            shared_features = self.shared_features_network(x)

            start_time = time.time()

            total_loss = torch.zeros(1).to(self.device)

            idx = 0
            for i in range(self.K):
                # print('i, idx', i, idx)
                count = counts[i]
                linear_weight = self.linear_q_weights[i]
                pred = linear_weight(shared_features[idx:idx+count])
                loss = F.mse_loss(pred, returns[idx:idx+count].view(-1, 1), reduction='sum')
                idx += count
                # print('loss', loss.item())
                total_loss += loss
            
            # print('total_loss', total_loss)
            total_loss = total_loss / self.target_minibatch_size

            # import ipdb; ipdb.set_trace()

            total_loss_time += (time.time() - start_time)

            total_loss.backward()
            losses.append(total_loss.item())
            self.both_optim.step()

            if _ % 250 == 0:
                print(total_loss.item())

                # print('sample', total_sample_time, 'loss', total_loss_time)
        # print('sample time', total_sample_time, 'loss time', total_loss_time)
        return dict(losses=losses)

    def new_replay_buffer(self, replay_buffer, copy_latest=False):
        self.replay_buffers.append(replay_buffer)
        self.linear_q_weights.append(torch.nn.Linear(self.feature_size, 1, bias=self.bias).to(self.device))
        self.K += 1
        if copy_latest and len(self.linear_q_weights) > 1:
            self.linear_q_weights[-1].load_state_dict(self.linear_q_weights[-2].state_dict())

        if self.max_replay_buffers > 0 and self.K > self.max_replay_buffers:
            del self.replay_buffers[0]
            del self.linear_q_weights[0]
            self.K -= 1


        self.all_params = []
        for weight in self.linear_q_weights:
            for param in weight.parameters():
                self.all_params.append(param)
        self.both_optim=optim.Adam(list(self.shared_features_network.parameters()) + self.all_params, self.shared_lr)

    def add_to_replay_buffer(self, i, paths):
        self.replay_buffers[i].push_many_with_returns(paths)

    def q_value(self, state, action, time, target_network=False):
        x = self.featurize_state_action_time(state, action, time)
        if target_network:
            shared_features = self.target_shared_features_network(x)
            pred = self.target_policy_linear(shared_features)
        else:
            shared_features = self.shared_features_network(x)
            pred = self.policy_linear(shared_features)

        return pred
    
    def predict(self, s, a, t):
        
        # import ipdb; ipdb.set_trace()

        return self.q_value(s, a, t)        

    # TODO: make use_mu_approx, n_value hyperparameters
    def value(self, state, time, target_network=False, use_mu_approx=True, n_value=1):
        if use_mu_approx:
            action = self.policy.get_action_pytorch(state, mean_action=True).detach().to(self.device)
            return self.q_value(state, action, time, target_network=target_network)
        else:
            values = None
            for _ in range(n_value):
                action = self.policy.get_action_pytorch(state).detach().to(self.device)
                if values is None:
                    values = self.q_value(state, action, time, target_network=target_network)
                else:
                    values += self.q_value(state, action, time, target_network=target_network)
            return values / n_value

    def advantage(self, state, action, time, use_mu_approx=True, n_value=1):
        return self.q_value(state, action, time) - self.value(state, time, use_mu_approx=use_mu_approx, n_value=n_value)


    def compute_bellman_targets(self, samples):
        next_s = samples['next_observations']
        terminal = samples['is_terminal'].view(-1, 1)
        r = samples['rewards']
        t = samples['time']
        bootstrap = self.value(next_s, t + 1, target_network=True).detach().view(-1, 1)
        target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - terminal)
        return target

    def update_bellman_only_linear(self):
        all_losses = []
        for _ in range(self.num_bellman_iters):
            losses = self.fit_targets_linear()
            self.target_policy_linear.load_state_dict(self.policy_linear.state_dict())
            all_losses.append(losses)
        return all_losses

    def fit_targets_linear(self):
        print('fit_targets_linear')
        linear_optim = optim.Adam(self.policy_linear.parameters(), self.linear_lr)
        losses = []
        for _ in range(self.num_bellman_fit_iters):

            samples, _ = self.sample_replay_buffers_efficient(self.bellman_minibatch_size)
            targets = self.compute_bellman_targets(samples).detach()
            s = samples['observations']
            a = samples['actions']
            t = samples['time']

            linear_optim.zero_grad()
            preds = self.q_value(s, a, t)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            losses.append(loss.item())
            linear_optim.step()
        return losses

    def update_bellman_only_shared(self):
        all_losses = []
        for _ in range(self.num_bellman_iters):
            losses = self.fit_targets_shared()
            self.target_shared_features_network.load_state_dict(self.shared_features_network.state_dict())
            all_losses.append(losses)
        return all_losses

    def fit_targets_shared(self):
        shared_optim = optim.Adam(self.shared_features_network.parameters(), self.shared_lr)
        losses = []
        for _ in range(self.num_bellman_fit_iters):

            samples, _ = self.sample_replay_buffers_efficient(self.bellman_minibatch_size)
            targets = self.compute_bellman_targets(samples).detach()
            s = samples['observations']
            a = samples['actions']
            t = samples['time']

            shared_optim.zero_grad()
            preds = self.q_value(s, a, t)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            losses.append(loss.item())
            shared_optim.step()
        return losses

    def update_bellman_both(self):
        all_losses = []
        for _ in range(self.num_bellman_iters):
            losses = self.fit_targets_shared()
            self.target_shared_features_network.load_state_dict(self.shared_features_network.state_dict())
            self.target_policy_linear.load_state_dict(self.policy_linear.state_dict())
            all_losses.append(losses)
        return all_losses

    def fit_targets_both(self):
        both_optim = optim.Adam(list(self.shared_features_network.parameters()) + list(self.policy_linear.parameters()), self.both_bellman_lr)
        losses = []
        for _ in range(self.num_bellman_fit_iters):
            samples, _ = self.sample_replay_buffers(self.bellman_minibatch_size)
            targets = self.compute_bellman_targets(samples).detach()
            s = samples['observations']
            a = samples['actions']
            t = samples['time']

            both_optim.zero_grad()
            preds = self.q_value(s, a, t)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            losses.append(loss.item())
            both_optim.step()
        return losses

    def update_bellman_both_regularized(self):
        update_stats = dict(losses=[], bellman_losses=[], reg_losses=[])
        for _ in range(self.num_bellman_iters):
            losses, bellman_losses, reg_losses = self.fit_targets_both_regularized()
            self.target_shared_features_network.load_state_dict(self.shared_features_network.state_dict())
            self.target_policy_linear.load_state_dict(self.policy_linear.state_dict())
            update_stats['losses'].append(losses)
            update_stats['bellman_losses'].append(bellman_losses)
            update_stats['reg_losses'].append(reg_losses)
        return update_stats

    def fit_targets_both_regularized(self):
        print('fit_targets_both_regularized')
        all_optim = optim.Adam(list(self.shared_features_network.parameters()) +
            list(self.policy_linear.parameters()) +
            self.all_params, self.both_bellman_lr)

        # all_optim = optim.SGD(list(self.shared_features_network.parameters()) +
        #     list(self.policy_linear.parameters()) +
        #     self.all_params, self.both_bellman_lr)

        losses = []
        bellman_losses = []
        reg_losses = []

        sample_time = 0.0
        targets_time = 0.0
        bellman_loss_time = 0.0
        total_loss_time = 0.0
        other_time = 0.0

        for _ in range(self.num_bellman_fit_iters):
            start_time = time.time()
            # samples, policies = self.sample_replay_buffers(self.bellman_minibatch_size)
            samples, counts = self.sample_replay_buffers_efficient(self.bellman_minibatch_size)
            after_rb_time = time.time()
            targets = self.compute_bellman_targets(samples).detach()
            after_target_time = time.time()
            s = samples['observations']
            a = samples['actions']
            t = samples['time']
            returns = samples['returns']

            all_optim.zero_grad()
            preds = self.q_value(s, a, t)
            bellman_loss = F.mse_loss(preds, targets)
            after_bellman_time = time.time()

            x = self.featurize_state_action_time(s, a, t)
            
            shared_features = self.shared_features_network(x)
            total_loss = torch.zeros(1).to(self.device)

            idx = 0
            for i in range(self.K):
                count = counts[i]
                linear_weight = self.linear_q_weights[i]
                pred = linear_weight(shared_features[idx:idx+count])
                loss = F.mse_loss(returns[idx:idx+count].view(-1, 1), pred, reduction='sum')
                idx += count
                total_loss += loss
            
            total_loss = total_loss / self.target_minibatch_size
            
            after_total_time = time.time()

            loss = bellman_loss + self.total_loss_weight * total_loss
            loss.backward()
            all_optim.step()

            bellman_losses.append(bellman_loss.item())
            reg_losses.append(total_loss.item())
            losses.append(loss.item())

            if _ % 50 == 0:
                print('bellman', bellman_loss.item(), 'total_loss', total_loss.item())

            end_time = time.time()

            sample_time += (after_rb_time - start_time)
            targets_time += (after_target_time - after_rb_time)
            bellman_loss_time += (after_bellman_time - after_target_time)
            total_loss_time += (after_total_time - after_bellman_time)
            other_time += (end_time - after_total_time)
        
        # print('sample_time', sample_time)
        # print('targets_time', targets_time)
        # print('bellman_loss_time', bellman_loss_time)
        # print('total_loss_time', total_loss_time)
        # print('other_time', other_time)

        return losses, bellman_losses, reg_losses

    def set_weight_newest(self):
        self.policy_linear.load_state_dict(self.linear_q_weights[-1].state_dict())
        self.target_policy_linear.load_state_dict(self.linear_q_weights[-1].state_dict())