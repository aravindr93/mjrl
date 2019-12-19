import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DynamicsModel:
    def __init__(self, state_dim, act_dim, hidden_size=(64, 64), seed=123,
                 fit_lr=1e-3, fit_wd=0.0, device='cpu', activation='relu', **kwargs):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
        self.network = DynamicsNet(state_dim, act_dim, hidden_size, seed=seed)
        self.network = self.network.to(self.device)
        self.network.set_transformations()
        if activation == 'tanh':
            print("Using tanh activation")
            self.network.nonlinearity = torch.tanh
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.loss_fn = torch.nn.MSELoss()

    def to(self, device):
        self.network.to(device)

    def is_cuda(self):
        return True if next(self.network.parameters()).is_cuda else False

    def forward(self, s, a):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.network.forward(s, a)

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        s_next = self.network.forward(s, a)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def fit(self, s, a, s_next, fit_mb_size, fit_epochs):
        return fit_model(self.network, s, a, s_next, self.optimizer,
                         self.loss_fn, fit_mb_size, fit_epochs, set_transforms=True)


class DynamicsNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_sizes=(64,64),
                 s_mean = None,
                 s_sigma = None,
                 a_mean = None,
                 a_sigma = None,
                 out_mean = None,
                 out_sigma = None,
                 seed=123,
                 ):
        super(DynamicsNet, self).__init__()

        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.layer_sizes = (state_dim + act_dim, ) + hidden_sizes + (state_dim, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)

        # for param in list(self.parameters())[-2:]:  # only last layer
        #     param.data = 1e-2 * param.data

    def set_transformations(self, s_mean=None, s_sigma=None,
                            a_mean=None, a_sigma=None,
                            out_mean=None, out_sigma=None):

        if s_mean is None:
            self.s_mean     = torch.zeros(self.state_dim)
            self.s_sigma    = torch.ones(self.state_dim)
            self.a_mean     = torch.zeros(self.act_dim)
            self.a_sigma    = torch.ones(self.act_dim)
            self.out_mean   = torch.zeros(self.state_dim)
            self.out_sigma  = torch.ones(self.state_dim)
        elif type(s_mean) == torch.Tensor:
            self.s_mean, self.s_sigma = s_mean, s_sigma
            self.a_mean, self.a_sigma = a_mean, a_sigma
            self.out_mean, self.out_sigma = out_mean, out_sigma
        elif type(s_mean) == np.ndarray:
            self.s_mean     = torch.from_numpy(np.float32(s_mean))
            self.s_sigma    = torch.from_numpy(np.float32(s_sigma))
            self.a_mean     = torch.from_numpy(np.float32(a_mean))
            self.a_sigma    = torch.from_numpy(np.float32(a_sigma))
            self.out_mean   = torch.from_numpy(np.float32(out_mean))
            self.out_sigma  = torch.from_numpy(np.float32(out_sigma))
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

    def forward(self, s, a):
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize inputs
        s_in = (s - self.s_mean)/(self.s_sigma + 1e-6)
        a_in = (a - self.a_mean)/(self.a_sigma + 1e-6)
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        # de-normalize the output with state transformations
        # out = out * (self.out_sigma + 1e-8) + self.out_mean
        # add back the un-normalized state
        # network is thus forced to learn the residual which is easier
        out = s + out
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


def fit_model(nn_model, s, a, s_next, optimizer,
              loss_func, batch_size, epochs,
              set_transforms=True):
    """
    :param nn_model:        pytorch model of form sp_hat = f(s, a) (class)
    :param s:               state at time t
    :param a:               action at time t
    :param s_next:          state at time t+1
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :param set_transforms:  set the model transforms from data (bool)
    :return:
    """

    assert type(s) == type(a)
    assert type(s) == type(s_next)
    assert s.shape[0] == a.shape[0]
    assert s.shape[0] == s_next.shape[0]

    device = 'cuda' if next(nn_model.parameters()).is_cuda else 'cpu'

    if type(s) == np.ndarray:
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s_next = torch.from_numpy(s_next).float()

    s = s.to(device)
    a = a.to(device)
    s_next = s_next.to(device)

    if set_transforms:
        delta = s_next - s
        s_mean, s_sigma = torch.mean(s, dim=0), torch.std(s, dim=0)
        a_mean, a_sigma = torch.mean(a, dim=0), torch.std(a, dim=0)
        out_mean, out_sigma = torch.mean(delta, dim=0), torch.std(delta, dim=0)
        nn_model.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)

    num_samples = s.shape[0]
    epoch_losses = []
    for ep in tqdm(range(epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = 0.0
        num_steps = int(num_samples // batch_size)
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_s  = s[data_idx]
            batch_a  = a[data_idx]
            batch_sp = s_next[data_idx]
            optimizer.zero_grad()
            sp_hat   = nn_model.forward(batch_s, batch_a)
            loss = loss_func(sp_hat, batch_sp)
            loss.backward()
            optimizer.step()
            ep_loss += loss.to('cpu').data.numpy()
        epoch_losses.append(ep_loss * 1.0/num_steps)
    # print("Loss after 1 epoch = %f | Loss after %i epochs = %f" % (epoch_losses[0], epochs, epoch_losses[-1]))
    return epoch_losses
