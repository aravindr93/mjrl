
from mjrl.baselines.baselines import ReplayBufferBaseline
from mjrl.utils.TupleMLP import TupleMLP
from mjrl.utils.optimize_model import fit_data

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import copy

import time

# class MBBaselineNaive(ReplayBufferBaseline):

#     def __init__(self, dynamics_model, value_network, target_value_network, policy,
#             reward_func, replay_buffer, H, T, gamma,
#             time_dim=3, model_batch_size=128, num_model_fit_iters=300, model_lr=1e-3,
#             bellman_batch_size=64, num_bellman_fit_iters=100, num_bellman_iters=10):
#         """
#             In this version, we will assume all data in the RB is from the same policy
#             This allows us to use s, a, s' in the replay buffer instead of 
#             sampling new actions and using our learned dynamics model.
#             If the data were from mixed policies, the bellman equations would not be valid.
#             Will implement the above next after verifying that this works
#         """
#         super().__init__(replay_buffer)

#         self.dynamics_model = dynamics_model
#         self.value_network = value_network
#         self.target_value_network = target_value_network
#         self.policy = policy
#         self.reward_func = reward_func

#         self.H = H
#         self.T = T
#         self.gamma = gamma
#         self.time_dim = time_dim
#         self.model_batch_size = model_batch_size
#         self.num_model_fit_iters = num_model_fit_iters
#         self.bellman_batch_size = bellman_batch_size
#         self.num_bellman_fit_iters = num_bellman_fit_iters
#         self.num_bellman_iters = num_bellman_iters

#         # should we pass optimizer in instead?
#         self.value_optim = torch.optim.Adam(self.value_network.parameters(), 5e-3)

#     def update(self):
#         model_loss, valid_loss = self.update_model()
#         self.bellman_update()

#         return {
#             'model_loss': model_loss,
#             'model_valid_loss': valid_loss
#             }        

#     def update_model(self):
#         print('update model')

#         s = np.concatenate([p['observations'][:-1] for p in self.replay_buffer.data_buffer])
#         a = np.concatenate([p['actions'][:-1] for p in self.replay_buffer.data_buffer])
#         sp = np.concatenate([p['observations'][1:] for p in self.replay_buffer.data_buffer])

#         num_valid = 500
#         valid_freq = 1

#         shuffle_idx = np.random.permutation(s.shape[0])
#         s_train = s[shuffle_idx][num_valid:]
#         a_train = a[shuffle_idx][num_valid:]
#         s_next_train = sp[shuffle_idx][num_valid:]

#         s_valid = s[shuffle_idx][:num_valid]
#         a_valid = a[shuffle_idx][:num_valid]
#         s_next_valid = sp[shuffle_idx][:num_valid]

#         model_loss, valid_loss = self.dynamics_model.fit_holdout(s_train, a_train, s_next_train,
#             s_valid, a_valid, s_next_valid, self.model_batch_size, self.num_model_fit_iters, valid_freq=valid_freq)
#         # model_loss = self.dynamics_model.fit(s, a, sp, self.model_batch_size, self.num_model_fit_iters)

#         return model_loss, valid_loss

#     def featurize_time(self, t):
#         if type(t) == np.ndarray:
#             t = torch.from_numpy(t).float()
#         t = t.float()
#         t = (t+1.0)/self.T
#         t_feat = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
#         return t_feat

#     def featurize_state_time(self, state, time):
#         # normalize time
#         t = self.featurize_time(time)
#         x = torch.cat([state, t], -1)
#         return x

#     def featurize_state_action(self, state, action):
#         x = torch.cat([state, action], -1)
#         return x

#     def value_no_model(self, state, time, target_network=False):
#         x = self.featurize_state_time(state, time)
#         if target_network:
#             return self.target_value_network(x)
#         else:
#             return self.value_network(x)

#     def value(self, state, time, target_network=False, stop=False):

#         s = state
#         value = np.zeros(state.size(0))
#         for t_offset in range(self.H):

#             # TODO: use mean action?
#             # TODO: Aravind: add noise, multiple rollouts averaged
#             a = self.policy.get_action_pytorch(s).detach()
#             # t = time + t_offset
            
#             value += self.gamma ** t_offset * self.reward_func(s.detach().numpy(), a.numpy()).reshape(-1)
            
#             # x = self.featurize_state_action(s, a)
#             # s = s + self.model_network(x)
#             s = self.dynamics_model.network(s, a)

#         # x = self.featurize_state_time(s, time + self.H)
#         if stop:
#             import ipdb; ipdb.set_trace()
#         return torch.from_numpy(value).float().view(-1, 1) + self.gamma**self.H * self.value_no_model(s, time+self.H, target_network=target_network).detach()


#     def compute_bellman_targets(self, idx, *args, **kwargs):
#         # For the states with index in the buffer given by idx, compute the Bellman targets
#         # If the index corresponds to a terminal state, then the value of next state is 0.0
#         amax = np.argmax(idx)
#         if idx[amax] == self.replay_buffer['observations'].shape[0]-1:
#             idx[(idx == idx[amax])] -= 1
#         next_s = self.replay_buffer['observations'][idx+1].detach()
#         terminal = self.replay_buffer['is_terminal'][idx].view(-1, 1).detach()
#         r = self.replay_buffer['rewards'][idx].detach()
#         t = self.replay_buffer['time'][idx].detach()
#         bootstrap = self.value(next_s, t + 1, target_network=True).detach().view(-1, 1)
#         target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - terminal)
#         return target

#     def fit_targets(self):
#         print('fit_targets')
#         for _ in range(self.num_bellman_fit_iters):
#             n = self.replay_buffer['observations'].shape[0]
#             idx = np.random.permutation(n)[:self.bellman_batch_size]
#             targets = self.compute_bellman_targets(idx).detach()
#             s = self.replay_buffer['observations'][idx]
#             t = self.replay_buffer['time'][idx]

#             x = self.featurize_state_time(s, t)

#             preds = self.value_network(x)
#             self.value_optim.zero_grad()
#             loss = F.mse_loss(preds, targets)
#             loss.backward()
#             self.value_optim.step()

#             if _ % 30 == 0:
#                 print('loss', loss.item())

#     def bellman_update(self):
#         for _ in range(self.num_bellman_iters):
#             self.fit_targets()
#             self.target_value_network.load_state_dict(self.value_network.state_dict())


class MBBaselineDoubleV(ReplayBufferBaseline):

    def __init__(self, fitted_model, value_network, no_update_value_network, policy,
            reward_func, replay_buffer, H, T, gamma,
            time_dim=3, lr=1e-3,
            batch_size=64, num_bellman_fit_iters=100, num_bellman_iters=10,
            reward_weight=1.0, value_weight=1.0):

        super().__init__(replay_buffer)

        self.fitted_model = fitted_model
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


        fitted_params = []
        for model in self.fitted_model:
            for param in model.network.parameters():
                fitted_params.append(param)
        # should we pass optimizer in instead?
        self.shared_optim = torch.optim.Adam(list(self.value_network.parameters())
            + fitted_params, lr)

        self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=1e-3)
    
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
        # print('update')
        all_stats = []
        for _ in range(self.num_bellman_iters):
            # all_stats.append(self.fit_custom_loss_max())
            all_stats.append(self.fit_custom_loss_all()) # TODO: max or all? ALL! at least for only learning reward
            self.no_update_value_network.load_state_dict(self.value_network.state_dict())
        return all_stats

    def update_traj(self):
        # print('update_traj')
        all_stats = []
        for _ in range(self.num_bellman_iters):
            all_stats.append(self.fit_traj_loss())
            self.no_update_value_network.load_state_dict(self.value_network.state_dict())
        return all_stats
    
    def update_all_models(self, mb_size=64, epochs=10):
        all_epoch_losses = []
        for model in self.fitted_model:
            s = np.concatenate([p['observations'][:-1] for p in self.replay_buffer.data_buffer])
            a = np.concatenate([p['actions'][:-1] for p in self.replay_buffer.data_buffer])
            sp = np.concatenate([p['observations'][1:] for p in self.replay_buffer.data_buffer])
            epoch_losses = model.fit(s, a, sp, mb_size, epochs)
            all_epoch_losses.append(epoch_losses)
        return all_epoch_losses

    def fit_returns(self, paths, mb_size=64, epochs=10):

        # make xs
        s = torch.from_numpy(np.concatenate([p['observations'] for p in paths])).to(self.policy.device).float()
        t = torch.from_numpy(np.concatenate([p['time'] for p in paths])).to(self.policy.device).float()
        x = self.featurize_state_time(s, t)
        # make ys
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        y = torch.from_numpy(returns).float().to(self.policy.device)

        ep_loss = fit_data(self.value_network, x, y, self.value_optim, F.mse_loss, mb_size, epochs)
        self.no_update_value_network.load_state_dict(self.value_network.state_dict())

        return [x.item() for x in ep_loss]
    
    def fit_on_policy(self, paths, mb_size, iters):

        # make xs
        s_all = torch.from_numpy(np.concatenate([p['observations'][:-1] for p in paths])).to(self.policy.device).float()
        a_all = torch.from_numpy(np.concatenate([p['actions'][:-1] for p in paths])).to(self.policy.device).float()
        s_next_all = torch.from_numpy(np.concatenate([p['observations'][1:] for p in paths])).to(self.policy.device).float()
        t_all = torch.from_numpy(np.concatenate([p['time'][:-1] for p in paths])).to(self.policy.device).float()
        # make ys
        returns = np.concatenate([path["returns"][:-1] for path in paths]).reshape(-1, 1)
        R_all = torch.from_numpy(returns).float().to(self.policy.device)
        R_next_all = torch.from_numpy(np.concatenate([path["returns"][1:] for path in paths]).reshape(-1, 1)).to(self.policy.device).float()

        N = s_all.shape[0]

        stats = dict(sum_total=0.0, sum_recon=0.0, sum_reward=0.0, sum_value=0.0)

        for i in range(iters):
            total_losses = []
            recon_losses = []
            reward_losses = []
            value_losses = []
            for model in self.fitted_model:
                mb_idx = np.random.choice(N, size=mb_size)
                s = s_all[mb_idx]
                a = a_all[mb_idx]
                s_next = s_next_all[mb_idx]
                t = t_all[mb_idx]

                R = R_all[mb_idx]
                R_next = R_next_all[mb_idx]

                s_next_hat = model.network(s,a)

                a_next = self.policy.get_action_pytorch(s_next) #TODO: Noise?
                # construct losses
                reconstruction_loss = F.mse_loss(s_next_hat, s_next)
                if self.reward_weight != 0.0:
                    reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next), self.reward_func(s_next, a_next))
                else:
                    reward_loss = torch.zeros(())

                #TODO: try below as well
                # a_next_hat = self.policy.get_action_pytorch(s_next_hat)
                # reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next_hat), self.reward_func(s_next, a_next))

                if self.value_weight != 0.0:
                    xp = self.featurize_state_time(s_next_hat, t+1)
                    V_hat = self.value_network(xp)
                    value_loss = F.mse_loss(V_hat, R_next)
                else:
                    value_loss = torch.zeros(())

                self.shared_optim.zero_grad()
                total_loss = reconstruction_loss + self.reward_weight * reward_loss + self.value_weight * value_loss
                total_losses.append(total_loss)

                recon_losses.append(reconstruction_loss)
                reward_losses.append(reward_loss)
                value_losses.append(value_loss)

            max_loss = np.argmax(total_losses)
            loss = total_losses[max_loss]
            loss.backward()

            stats['sum_total'] += loss.item()
            stats['sum_recon'] += recon_losses[max_loss].item()
            stats['sum_reward'] += reward_losses[max_loss].item()
            stats['sum_value'] += value_losses[max_loss].item()

            stats['last_total'] = loss.item()
            stats['last_recon'] = recon_losses[max_loss].item()
            stats['last_reward'] = reward_losses[max_loss].item()
            stats['last_value'] = value_losses[max_loss].item()

            self.shared_optim.step()

        self.no_update_value_network.load_state_dict(self.value_network.state_dict())

        for k in stats.keys():
            stats[k] /= iters

        return stats

    def fit_custom_loss_max(self):
        # print('fit_custom_loss_max')

        stats = dict(sum_total=0.0, sum_recon=0.0, sum_reward=0.0, sum_value=0.0, used_models=[])   

        for i in range(self.num_bellman_fit_iters):
            # get samples
            samples = self.replay_buffer.get_sample_safe(self.batch_size)
            s = samples['observations'].detach()
            a = samples['actions'].detach()
            s_next = samples['next_observations'].detach()
            t = samples['time'].detach()


            total_losses = []
            recon_losses = []
            reward_losses = []
            value_losses = []

            for model in self.fitted_model:
                s_next_hat = model.network(s,a)

                a_next = self.policy.get_action_pytorch(s_next) #TODO: Noise?
                # construct losses
                reconstruction_loss = F.mse_loss(s_next_hat, s_next)
                if self.reward_weight != 0.0:
                    reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next), self.reward_func(s_next, a_next))
                else:
                    reward_loss = torch.zeros(())

                #TODO: try below as well
                # a_next_hat = self.policy.get_action_pytorch(s_next_hat)
                # reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next_hat), self.reward_func(s_next, a_next))

                if self.value_weight != 0.0:
                    x = self.featurize_state_time(s, t)
                    xp = self.featurize_state_time(s_next_hat, t+1)
                    V_hat = self.value_network(x)
                    target = self.reward_func(s, a).view(-1, 1) + self.gamma * self.value_network(xp)
                    value_loss = F.mse_loss(V_hat, target)
                else:
                    value_loss = torch.zeros(())

                self.shared_optim.zero_grad()
                total_loss = reconstruction_loss + self.reward_weight * reward_loss + self.value_weight * value_loss
                total_losses.append(total_loss)

                recon_losses.append(reconstruction_loss)
                reward_losses.append(reward_loss)
                value_losses.append(value_loss)

            max_loss = np.argmax(total_losses)
            loss = total_losses[max_loss]
            loss.backward()

            stats['sum_total'] += loss.item()
            stats['sum_recon'] += recon_losses[max_loss].item()
            stats['sum_reward'] += reward_losses[max_loss].item()
            stats['sum_value'] += value_losses[max_loss].item()
            stats['used_models'].append(max_loss)

            self.shared_optim.step()

            # if i % 25 == 0:
            #     print('recon', reconstruction_loss.item(), 'rew', reward_loss.item(), 'value', value_loss.item())
            #     print('max loss model', max_loss)
        
        for k in stats.keys():
            if k.startswith('sum'):
                stats[k] /= self.num_bellman_fit_iters

        return stats

    def fit_custom_loss_all(self):
        # print('fit_custom_loss_all')

        stats = dict(sum_total=0.0, sum_recon=0.0, sum_reward=0.0, sum_value=0.0, used_models=[])   

        sample_time = 0.0
        recon_loss_time = 0.0
        reward_time = 0.0
        value_time = 0.0
        construct_loss_time = 0.0
        backward_time = 0.0

        for i in range(self.num_bellman_fit_iters):

            total_losses = []
            recon_losses = []
            reward_losses = []
            value_losses = []

            for model in self.fitted_model:
                # get samples
                start = time.time()
                samples = self.replay_buffer.get_sample_safe(self.batch_size)
                s = samples['observations'].detach()
                a = samples['actions'].detach()
                s_next = samples['next_observations'].detach()
                t = samples['time'].detach()
                
                sample_time += time.time() - start
                
                start = time.time()
                s_next_hat = model.network(s,a)

                a_next = self.policy.get_action_pytorch(s_next) #TODO: Noise?
                # construct losses
                reconstruction_loss = F.mse_loss(s_next_hat, s_next)

                recon_loss_time += time.time() - start

                start = time.time()
                if self.reward_weight != 0.0:
                    reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next), self.reward_func(s_next, a_next))
                else:
                    reward_loss = torch.zeros(())

                reward_time += time.time() - start
                #TODO: try below as well
                # a_next_hat = self.policy.get_action_pytorch(s_next_hat)
                # reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next_hat), self.reward_func(s_next, a_next))

                start = time.time()
                if self.value_weight != 0.0:
                    x = self.featurize_state_time(s, t)
                    xp = self.featurize_state_time(s_next_hat, t + 1)  # NEW: add detach here?
                                                                        # TODO: try average value over models (same state)
                    V_hat = self.value_network(x)
                    target = self.reward_func(s, a).view(-1, 1) + self.gamma * self.value_network(xp)
                    # target = self.reward_func(s, a).view(-1, 1) + self.gamma * self.no_update_value_network(xp)
                    value_loss = F.mse_loss(V_hat, target)
                else:
                    value_loss = torch.zeros(())

                value_time += time.time() - start

                start = time.time()
                self.shared_optim.zero_grad()
                total_loss = reconstruction_loss + self.reward_weight * reward_loss + self.value_weight * value_loss
                total_losses.append(total_loss)

                recon_losses.append(reconstruction_loss)
                reward_losses.append(reward_loss)
                value_losses.append(value_loss)

                construct_loss_time += time.time() - start

            start = time.time()
            all_recon_loss = torch.stack(recon_losses).sum()
            all_reward_loss = torch.stack(reward_losses).sum()
            all_value_loss = torch.stack(value_losses).sum()
            
            all_model_loss = torch.stack(total_losses).sum()
            all_model_loss.backward()
            backward_time += time.time() - start
            stats['sum_total'] += all_model_loss.item() / len(self.fitted_model)
            stats['sum_recon'] += all_recon_loss.item() / len(self.fitted_model)
            stats['sum_reward'] += all_reward_loss.item() / len(self.fitted_model)
            stats['sum_value'] += all_value_loss.item() / len(self.fitted_model)

            self.shared_optim.step()

            # if i % 25 == 0:
            #     print('recon', reconstruction_loss.item(), 'rew', reward_loss.item(), 'value', value_loss.item())
            #     print('max loss model', max_loss)
        
        for k in stats.keys():
            if k.startswith('sum'):
                stats[k] /= self.num_bellman_fit_iters

        # print('sample_time', sample_time)
        # print('recon_loss_time', recon_loss_time)
        # print('reward_time', reward_time)
        # print('value_time', value_time)
        # print('construct_loss_time', construct_loss_time)
        # print('backward_time', backward_time)

        return stats

    def fit_traj_loss(self):
        # sample a trajectory

        stats = dict(sum_total=0.0, sum_recon=0.0, sum_reward=0.0, sum_value=0.0)

        for i in range(self.num_bellman_fit_iters):
            traj_idx = np.random.choice(len(self.replay_buffer.data_buffer))
            path = self.replay_buffer.data_buffer[traj_idx]

            s = torch.from_numpy(path['observations'][:-1]).to(self.policy.device).float().detach()
            a = torch.from_numpy(path['actions'][:-1]).to(self.policy.device).float().detach()
            s_next = torch.from_numpy(path['observations'][1:]).to(self.policy.device).float().detach()
            t = torch.from_numpy(path['time'][:-1]).to(self.policy.device).float().detach()

            total_losses = []
            recon_losses = []
            reward_losses = []
            value_losses = []
            for model in self.fitted_model:
                s_next_hat = model.network(s,a)

                a_next = self.policy.get_action_pytorch(s_next) #TODO: Noise?
                # construct losses
                reconstruction_loss = F.mse_loss(s_next_hat, s_next)
                if self.reward_weight != 0.0:
                    reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next), self.reward_func(s_next, a_next))
                else:
                    reward_loss = torch.zeros(())

                #TODO: try below as well
                # a_next_hat = self.policy.get_action_pytorch(s_next_hat)
                # reward_loss = F.mse_loss(self.reward_func(s_next_hat, a_next_hat), self.reward_func(s_next, a_next))

                if self.value_weight != 0.0:
                    x = self.featurize_state_time(s, t)
                    xp = self.featurize_state_time(s_next_hat, t+1)
                    V_hat = self.value_network(x)
                    target = self.reward_func(s, a).view(-1, 1) + self.gamma * self.value_network(xp)
                    value_loss = F.mse_loss(V_hat, target)
                else:
                    value_loss = torch.zeros(())

                self.shared_optim.zero_grad()
                total_loss = reconstruction_loss + self.reward_weight * reward_loss + self.value_weight * value_loss
                total_losses.append(total_loss)

                recon_losses.append(reconstruction_loss)
                reward_losses.append(reward_loss)
                value_losses.append(value_loss)

            max_loss = np.argmax(total_losses)
            loss = total_losses[max_loss]
            loss.backward()

            # sum_loss += loss.item()

            stats['sum_total'] += loss.item()
            stats['sum_recon'] += recon_losses[max_loss].item()
            stats['sum_reward'] += reward_losses[max_loss].item()
            stats['sum_value'] += value_losses[max_loss].item()

            self.shared_optim.step()

            # if i % 25 == 0:
            #     print('recon', reconstruction_loss.item(), 'rew', reward_loss.item(), 'value', value_loss.item())
            #     print('max loss model', max_loss)
        
        for k in stats.keys():
            stats[k] /= self.num_bellman_fit_iters

        return stats


        # compute the loss

        # update
        pass

    def value_no_model(self, state, time):
        x = self.featurize_state_time(state, time)
        return self.value_network(x)
    
    def predict(self, path):
        obs = torch.from_numpy(path['observations']).float().to(self.policy.device)
        t = torch.from_numpy(path['time']).float().to(self.policy.device)
        return self.value_no_model(obs, t).cpu().detach().numpy().ravel()

    def value(self, state, time, use_average=False, n_rollouts=4, add_tvf=True):
        if use_average:
            s = state
            value = np.zeros(state.size(0))
            for t_offset in range(self.H):

                a = self.policy.get_action_pytorch(s, mean_action=True).detach()
                # t = time + t_offset
                
                value += self.gamma ** t_offset * self.reward_func(s.detach().cpu().numpy(), a.cpu().numpy()).reshape(-1)

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
                    
                    value += self.gamma ** t_offset * self.reward_func(s.detach().cpu().numpy(), a.cpu().numpy()).reshape(-1)

                    s = self.dynamics_model.network(s, a)

                if add_tvf:
                    single_val = torch.from_numpy(value).float().view(-1, 1) + self.gamma ** self.H * self.value_no_model(s, time + self.H).detach()
                else:
                    single_val = torch.from_numpy(value).float().view(-1, 1)
            total_value += single_val
            return total_value / n_rollouts
    
    def model_value(self, state, time, rollouts_per_model=1, add_tvf=True):
        # rollout all models, average

        total_value = torch.zeros(state.size(0), 1).to(self.policy.device)
        
        for model in self.fitted_model:
            for rollout in range(rollouts_per_model):
                s = state
                value = np.zeros(state.size(0))
                for t_offset in range(self.H):
                    a = self.policy.get_action_pytorch(s, mean_action=False).detach()
                    value += self.gamma ** t_offset * self.reward_func(s.detach().cpu().numpy(), a.cpu().numpy()).reshape(-1)
                    s = model.network(s, a)

                single_val = torch.from_numpy(value).float().view(-1, 1).to(self.policy.device)

                if add_tvf:
                    single_val += self.gamma ** self.H * self.value_no_model(s, time + self.H).detach()

                total_value += single_val

        return total_value / (len(self.fitted_model) * rollouts_per_model)