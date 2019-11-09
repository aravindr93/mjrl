import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time


class QNetwork(nn.Module):
    def __init__(self, state_dim, act_dim, time_feat_dim=4,
                 hidden_sizes=(64,64),
                 transforms=None,
                 nonlineariry = None,
                 seed=123,
                 device='cpu'
                 ):
        super(QNetwork, self).__init__()

        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.t_feat_dim = time_feat_dim
        self.hidden_sizes = hidden_sizes
        self.layer_sizes = (state_dim + act_dim + time_feat_dim, ) + hidden_sizes + (1, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu if nonlineariry is None else nonlineariry

        # transforms
        self.device = device
        self.reset_transforms()
        self.set_transforms(transforms)

        for param in list(self.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data

        self.to(self.device)

    # def to(self, device, *args, **kwargs):
    #     # device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
    #     print('to() device', device)
    #     self.device = 'cpu' if device is None else device
    #     self = super().to(*args, **kwargs)
    #     self.transforms_to()
    #     self.transforms = self.make_transforms_dict()
    #     return self

    def to(self, device):
        self.transforms_to()
        self=super().to(device)
        return self

    def make_transforms_dict(self):
        transforms = dict(s_mean=self.s_mean, s_sigma=self.s_sigma,
                          a_mean=self.a_mean, a_sigma=self.a_sigma,
                          t_feat_mean=self.t_feat_mean, t_feat_sigma=self.t_feat_sigma,
                          out_mean=self.out_mean, out_sigma=self.out_sigma
                          )
        return transforms



    def reset_transforms(self):
        self.s_mean = torch.zeros(self.state_dim)
        self.s_sigma = torch.ones(self.state_dim)
        self.a_mean = torch.zeros(self.act_dim)
        self.a_sigma = torch.ones(self.act_dim)
        self.t_feat_mean = torch.zeros(self.t_feat_dim)
        self.t_feat_sigma = torch.ones(self.t_feat_dim)
        self.out_mean = torch.tensor(0.0)
        self.out_sigma = torch.tensor(1.0)

        self.transforms_to()

        self.transforms = self.make_transforms_dict()

    def transforms_to(self, device=None):
        if device is not None:
            self.device = device

        self.s_mean = self.s_mean.to(self.device)
        self.s_sigma = self.s_sigma.to(self.device)
        self.a_mean = self.a_mean.to(self.device)
        self.a_sigma = self.a_sigma.to(self.device)
        self.t_feat_mean =  self.t_feat_mean.to(self.device)
        self.t_feat_sigma = self.t_feat_sigma.to(self.device)
        self.out_mean = self.out_mean.to(self.device)
        self.out_sigma = self.out_sigma.to(self.device)
        

    def set_transforms(self, transforms=None):
        # transforms is either None or of type dictionary
        transforms = None if (transforms == dict() or transforms is None) else transforms
        if transforms is not None:
            dtype_tensor = True if type(transforms['s_mean']) == torch.Tensor else False
            self.s_mean = transforms['s_mean'] if dtype_tensor else torch.tensor(transforms['s_mean']).float()
            self.s_sigma = transforms['s_sigma'] if dtype_tensor else torch.tensor(transforms['s_sigma']).float()
            self.a_mean = transforms['a_mean'] if dtype_tensor else torch.tensor(transforms['a_mean']).float()
            self.a_sigma = transforms['a_sigma'] if dtype_tensor else torch.tensor(transforms['a_sigma']).float()
            self.t_feat_mean = transforms['t_feat_mean'] if dtype_tensor else torch.tensor(transforms['t_feat_mean']).float()
            self.t_feat_sigma = transforms['t_feat_sigma'] if dtype_tensor else torch.tensor(transforms['t_feat_sigma']).float()
            self.out_mean = transforms['out_mean'] if dtype_tensor else torch.tensor(transforms['out_mean']).float()
            self.out_sigma = transforms['out_sigma'] if dtype_tensor else torch.tensor(transforms['out_sigma']).float()

        else:
            print("Not setting the transforms. Input was either None or in unsupported format.")
        self.transforms_to()
        self.transforms = self.make_transforms_dict()

    def forward(self, s, a, t_feat):
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        s = s.to(self.device)
        a = a.to(self.device)

        # normalize inputs
        s_in = (s - self.s_mean)/(self.s_sigma + 1e-6)
        a_in = (a - self.a_mean)/(self.a_sigma + 1e-6)
        t_in = (t_feat - self.t_feat_mean)/(self.t_feat_sigma + 1e-6)
        out = torch.cat([s_in, a_in, t_in], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        # de-normalize the output with state transformations
        out = out * self.out_sigma + self.out_mean
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = dict(s_mean=self.s_mean, s_sigma=self.s_sigma,
                          a_mean=self.a_mean, a_sigma=self.a_sigma,
                          t_feat_mean=self.t_feat_mean, t_feat_sigma=self.t_feat_sigma,
                          out_mean=self.out_mean, out_sigma=self.out_sigma
                          )
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transforms(new_params['transforms'])


class QPi:
    def __init__(self, policy, state_dim, act_dim, time_dim, horizon, replay_buffer, gamma=0.9, hidden_size=(64, 64), seed=123,
                 fit_lr=1e-3, fit_wd=0.0, batch_size=64, num_bellman_iters=1, num_fit_iters=16, device='cpu', activation='relu',
                 use_mu_approx=True, num_value_actions=-1, summary_writer=None, **kwargs):

        # Terminology:
        # Bellman iterations : (outer loop) in each iteration, we sync the target network and learner network
        #                      and perform many gradient steps to make learner network approximate target
        # Fit iterations :     (inner loop) we perform many gradient steps to approximate the target here

        self.policy = policy
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.time_dim = time_dim
        self.horizon = horizon
        self.gamma = gamma
        self.device = 'cuda' if (device == 'gpu' or device == 'cuda') else 'cpu'
        self.buffer = replay_buffer.to(self.device)

        self.network = QNetwork(state_dim, act_dim, time_dim, hidden_size, seed=seed, device=device)
        self.network.to(self.device)

        self.target_network = QNetwork(state_dim, act_dim, time_dim, hidden_size, seed=seed, device=device)
        self.target_network = self.target_network.to(self.device)
        self.target_network.set_params(self.network.get_params())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.loss_fn = torch.nn.MSELoss()

        self.batch_size = batch_size                # batch size for each gradient step
        self.num_fit_iters = num_fit_iters          # fixing targets, number of optimization steps to approximate target
        self.num_bellman_iters = num_bellman_iters  # number of times to sync target and learner networks

        self.use_mu_approx = use_mu_approx
        self.num_value_actions = num_value_actions
        if not self.use_mu_approx:
            if self.num_value_actions < 0:
                raise ValueError('num_value_actions must be greater than 0 when use_mu_approx is False. got: {}'.format(self.num_value_actions))
        
        self.summary_writer = summary_writer

    def to(self, device):
        self.network.to(device)
        self.target_network.to(device)

    def is_cuda(self):
        return True if next(self.network.parameters()).is_cuda else False

    def forward(self, s, a, t, target_network=False):
        # here t is a 1D array, which we featurize and pass to Q network.
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        t_feat = self.featurize_time(t)
        s = s.to(self.device)
        a = a.to(self.device)
        t_feat = t_feat.to(self.device)
        if target_network:
            prediction = self.target_network.forward(s, a, t_feat)
        else:
            prediction = self.network.forward(s, a, t_feat)
        return prediction

    def featurize_time(self, t):
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).float()
        t = t.to(self.device)
        t = t.float()
        t = (t+1.0)/self.horizon
        t_feat = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
        return t_feat

    def predict(self, s, a, t, target_network=False):
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if type(t) != np.ndarray:
            # t = np.array([t])
            t = t.detach().cpu().numpy()
        
        Q = self.forward(s, a, t, target_network)
        return Q.to('cpu').data.numpy()

    def compute_average_value(self, s, t, target_network=False):
        if self.use_mu_approx:
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            s = s.to('cpu')
            mu = self.policy.model.forward(s)
            Q = self.forward(s, mu, t, target_network)
            return [mu, Q, Q]
        else:
            # to return: (actions, corresponding Qs, and mean)
            n = s.shape[0]
            values = torch.zeros((n, 1), device=self.device)
            actions = []
            Qss = []
            for _ in range(self.num_value_actions):
                if type(s) == torch.Tensor: # TODO: keep as tensor. requires some code refactor
                    s = s.detach().cpu().numpy()

                a = self.policy.get_action_batch(s)
                Qs = self.forward(s, a, t, target_network=target_network)
                values += Qs
                actions.append(a)
                Qss.append(Qs)
            values /= self.num_value_actions

            return [actions, Qss, values]

    def compute_bellman_targets(self, idx, *args, **kwargs):
        # For the states with index in the buffer given by idx, compute the Bellman targets
        # If the index corresponds to a terminal state, then the value of next state is 0.0
        amax = np.argmax(idx)
        if idx[amax] == self.buffer['observations'].shape[0]-1:
            idx[(idx == idx[amax])] -= 1
        next_s = self.buffer['observations'][idx+1].detach()
        terminal = self.buffer['is_terminal'][idx].view(-1, 1).detach()
        r = self.buffer['rewards'][idx].detach()
        t = self.buffer['time'][idx].detach()
        bootstrap = self.compute_average_value(next_s, t+1, target_network=True)[-1].detach()
        target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - terminal)
        return target

    def fit_targets(self):
        losses = []
        for _ in range(self.num_fit_iters):
            n = self.buffer['observations'].shape[0]
            idx = np.random.permutation(n)[:self.batch_size]
            targets = self.compute_bellman_targets(idx).detach()
            s = self.buffer['observations'][idx]
            a = self.buffer['actions'][idx]
            t = self.buffer['time'][idx]
            preds = self.forward(s, a, t)
            self.optimizer.zero_grad()
            loss = self.loss_fn(preds, targets)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return losses

    def bellman_update(self):
        # This function should perform the bellman updates (i.e. fit targets, sync networks, fit targets again ...)
        losses = []
        start = time.time()
        for bellman_iter in tqdm(range(self.num_bellman_iters)):
            # sync the learner and target networks
            self.target_network.set_params(self.network.get_params())
            # make network approximate Bellman targets
            iter_loss = self.fit_targets()
            losses.append(np.mean(iter_loss))
        update_time = time.time() - start
        return losses, update_time
