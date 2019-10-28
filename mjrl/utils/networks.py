import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, state_dim, act_dim, time_dim=4,
                 hidden_sizes=(64,64),
                 s_mean = None,
                 s_sigma = None,
                 a_mean = None,
                 a_sigma = None,
                 out_mean = None,
                 out_sigma = None,
                 nonlineariry = None,
                 seed=123,
                 ):
        super(QNetwork, self).__init__()

        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.time_dim = time_dim
        self.hidden_sizes = hidden_sizes
        self.layer_sizes = (state_dim + act_dim + time_dim, ) + hidden_sizes + (1, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu if nonlineariry is None else nonlineariry
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)

        for param in list(self.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data

    def set_transformations(self, s_mean=None, s_sigma=None,
                            a_mean=None, a_sigma=None,
                            out_mean=None, out_sigma=None):

        if s_mean is None:
            self.s_mean     = torch.zeros(self.state_dim)
            self.s_sigma    = torch.ones(self.state_dim)
            self.a_mean     = torch.zeros(self.act_dim)
            self.a_sigma    = torch.ones(self.act_dim)
            self.out_mean   = torch.tensor(0.0)
            self.out_sigma  = torch.tensor(1.0)
        elif type(s_mean) == torch.Tensor:
            self.s_mean, self.s_sigma = s_mean, s_sigma
            self.a_mean, self.a_sigma = a_mean, a_sigma
            self.out_mean, self.out_sigma = out_mean, out_sigma
        elif type(s_mean) == np.ndarray:
            self.s_mean     = torch.tensor(s_mean).float()
            self.s_sigma    = torch.tensor(s_sigma).float()
            self.a_mean     = torch.tensor(a_mean).float()
            self.a_sigma    = torch.tensor(a_sigma).float()
            self.out_mean   = torch.tensor(out_mean).float()
            self.out_sigma  = torch.tensor(out_sigma).float()
        else:
            print("Unknown type for transformations")
            quit()

        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        self.s_mean, self.s_sigma = self.s_mean.to(device), self.s_sigma.to(device)
        self.a_mean, self.a_sigma = self.a_mean.to(device), self.a_sigma.to(device)
        self.out_mean, self.out_sigma = self.out_mean.to(device), self.out_sigma.to(device)

        self.transformations = dict(s_mean=self.s_mean, s_sigma=self.s_sigma,
                                    a_mean=self.a_mean, a_sigma=self.a_sigma,
                                    out_mean=self.out_mean, out_sigma=self.out_sigma)

    def forward(self, s, a, t):
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize inputs
        s_in = (s - self.s_mean)/(self.s_sigma + 1e-6)
        a_in = (a - self.a_mean)/(self.a_sigma + 1e-6)
        t_in = t
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
        transforms = (self.s_mean, self.s_sigma,
                      self.a_mean, self.a_sigma,
                      self.out_mean, self.out_sigma)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)


class QPi:
    def __init__(self, policy, state_dim, act_dim, time_dim, horizon, replay_buffer, gamma=0.9, hidden_size=(64, 64), seed=123,
                 fit_lr=1e-3, fit_wd=0.0, device='cpu', activation='relu', **kwargs):
        self.policy = policy
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.time_dim = time_dim
        self.horizon = horizon
        self.gamma = gamma
        self.buffer = replay_buffer
        self.device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
        self.network = QNetwork(state_dim, act_dim, time_dim, hidden_size, seed=seed)
        self.network = self.network.to(self.device)
        self.network.set_transformations()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.loss_fn = torch.nn.MSELoss()

    def to(self, device):
        self.network.to(device)

    def is_cuda(self):
        return True if next(self.network.parameters()).is_cuda else False

    def forward(self, s, a, t):
        # here t is a 1D array, which we featurize and pass to Q network.
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        t = self.featurize_time(t)
        s = s.to(self.device)
        a = a.to(self.device)
        t = t.to(self.device)
        return self.network.forward(s, a, t)

    # TODO(Aravind): Clearly distinguish time from featurized time

    def featurize_time(self, t):
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).float()
        t = t.float()
        t = t/self.horizon
        t = torch.stack([t**(k+1) for k in range(self.time_dim)], -1)
        return t

    def predict(self, s, a, t):
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if type(t) != np.ndarray:
            t = np.array([t])
        Q = self.forward(s, a, t)
        return Q.to('cpu').data.numpy()

    def compute_average_value(self, s, t, use_mu_approx=True):
        if use_mu_approx:
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            s = s.to('cpu')
            mu = self.policy.model.forward(s)
            Q = self.forward(s, mu, t)
            return [mu, Q, Q]
        else:
            # to return: (actions, corresponding Qs, and mean)
            raise NotImplementedError

    def compute_bellman_targets(self, idx, use_mu_approx=True, *args, **kwargs):
        amax = torch.argmax(idx)
        if idx[amax] == self.buffer['observations'].shape[0]:
            idx[(idx == idx[amax])] -= 1
        next_s = self.buffer['next_observations'][idx+1]
        terminal = self.buffer['is_terminal'][idx]
        r = torch.tensor(buffer['rewards']).float()
        t = self.buffer['time'][idx]
        bootstrap = self.compute_average_value(next_s, t, use_mu_approx)[-1]
        target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - torch.tensor(terminal).float().view(-1, 1))
        return target

    def bellman_update(self, mini_buffer, fit_mb_size, num_grad_steps=100):
        # this should be where the fitting level optimization happens
        s = mini_buffer['observations']
        a = mini_buffer['actions']
        r = torch.tensor(mini_buffer['rewards']).float()
        t = mini_buffer['time']
        next_s = mini_buffer['next_observations']
        terminal = mini_buffer['is_terminal']
        bootstrap = self.compute_average_value(next_s, t)[-1]
        target = r.view(-1, 1) + self.gamma * bootstrap * (1.0 - torch.tensor(terminal).float().view(-1, 1))
        t_feat = self.featurize_time(t)
        return fit_model(self.network, [s, a, t_feat], target.detach(),
                         self.optimizer, self.loss_fn, fit_mb_size,
                         set_transforms=False, num_grad_steps=num_grad_steps)


    def update_network(self):
        # make fit part of class
        pass


def fit_model(nn_model, input_list, target, optimizer,
              loss_func, batch_size, epochs=1,
              set_transforms=True,
              target_callback=None,
              num_grad_steps=None):

    num_samples = input_list[0].shape[0]
    assert np.array([d.shape[0] == num_samples for d in input_list]).all()

    device = 'cuda' if next(nn_model.parameters()).is_cuda else 'cpu'

    for idx, d in enumerate(input_list):
        if type(d) == np.ndarray:
            input_list[idx] = torch.from_numpy(d).float()
        input_list[idx] = input_list[idx].to(device)

    if type(target) == np.ndarray:
        target = target.to(device)

    #TODO(Aravind): Implement transforms in a generic way

    if num_grad_steps is None:
        epoch_losses = []
        for ep in tqdm(range(epochs)):
            rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
            ep_loss = 0.0
            num_steps = int(num_samples // batch_size)
            for mb in range(num_steps):
                data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
                batch_inp = [d[data_idx] for d in input_list]
                batch_tar = target[data_idx]
                optimizer.zero_grad()
                prediction = nn_model.forward(*batch_inp)
                optimizer.zero_grad()
                loss = loss_func(prediction, batch_tar)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            epoch_losses.append(ep_loss * 1.0/num_steps)
        return epoch_losses
    else:
        average_loss = 0.0
        for mb in range(num_grad_steps):
            data_idx = torch.LongTensor(np.random.choice(num_samples, batch_size))
            batch_inp = [d[data_idx] for d in input_list]
            batch_tar = target[data_idx]
            prediction = nn_model.forward(*batch_inp)
            optimizer.zero_grad()
            loss = loss_func(prediction, batch_tar)
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
        average_loss = average_loss / num_grad_steps
        return [average_loss]
