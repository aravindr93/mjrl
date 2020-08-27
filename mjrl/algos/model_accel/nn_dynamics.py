import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class WorldModel:
    def __init__(self, state_dim, act_dim,
                 learn_reward=False,
                 hidden_size=(64,64),
                 seed=123,
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu',
                 residual=True,
                 *args,
                 **kwargs):

        self.state_dim, self.act_dim = state_dim, act_dim
        self.device, self.learn_reward = device, learn_reward
        if self.device == 'gpu' : self.device = 'cuda'
        # construct the dynamics model
        self.dynamics_net = DynamicsNet(state_dim, act_dim, hidden_size, residual=residual, seed=seed).to(self.device)
        self.dynamics_net.set_transformations()  # in case device is different from default, it will set transforms correctly
        if activation == 'tanh' : self.dynamics_net.nonlinearity = torch.tanh
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.dynamics_loss = torch.nn.MSELoss()
        # construct the reward model if necessary
        if self.learn_reward:
            # small network for reward is sufficient if we augment the inputs with next state predictions
            self.reward_net = RewardNet(state_dim, act_dim, hidden_size=(100, 100), seed=seed).to(self.device)
            self.reward_net.set_transformations()  # in case device is different from default, it will set transforms correctly
            if activation == 'tanh' : self.reward_net.nonlinearity = torch.tanh
            self.reward_opt = torch.optim.Adam(self.reward_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
            self.reward_loss = torch.nn.MSELoss()
        else:
            self.reward_net, self.reward_opt, self.reward_loss = None, None, None

    def to(self, device):
        self.dynamics_net.to(device)
        if self.learn_reward : self.reward_net.to(device)

    def is_cuda(self):
        return next(self.dynamics_net.parameters()).is_cuda

    def forward(self, s, a):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.dynamics_net.forward(s, a)

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        s_next = self.dynamics_net.forward(s, a)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def reward(self, s, a):
        if not self.learn_reward:
            print("Reward model is not learned. Use the reward function from env.")
            return None
        else:
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float()
            s = s.to(self.device)
            a = a.to(self.device)
            sp = self.dynamics_net.forward(s, a).detach().clone()
            return self.reward_net.forward(s, a, sp)

    def compute_loss(self, s, a, s_next):
        # Intended for logging use only, not for loss computation
        sp = self.forward(s, a)
        s_next = torch.from_numpy(s_next).float() if type(s_next) == np.ndarray else s_next
        s_next = s_next.to(self.device)
        loss = self.dynamics_loss(sp, s_next)
        return loss.to('cpu').data.numpy()

    def fit_dynamics(self, s, a, sp, fit_mb_size, fit_epochs, max_steps=1e4, 
                     set_transformations=True, *args, **kwargs):
        # move data to correct devices
        assert type(s) == type(a) == type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()
        s = s.to(self.device); a = a.to(self.device); sp = sp.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_mean, s_sigma = torch.mean(s, dim=0), torch.std(s, dim=0)
            a_mean, a_sigma = torch.mean(a, dim=0), torch.std(a, dim=0)
            out_mean, out_sigma = torch.mean(sp-s, dim=0), torch.std(sp-s, dim=0)
            self.dynamics_net.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)

        # call the generic fit function
        X = (s, a) ; Y = sp
        return fit_model(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    def fit_reward(self, s, a, r, fit_mb_size, fit_epochs, max_steps=1e4, 
                   set_transformations=True, *args, **kwargs):
        if not self.learn_reward:
            print("Reward model was not initialized to be learnable. Use the reward function from env.")
            return None

        # move data to correct devices
        assert type(s) == type(a) == type(r)
        assert len(r.shape) == 2 and r.shape[1] == 1  # r should be a 2D tensor, i.e. shape (N, 1)
        assert s.shape[0] == a.shape[0] == r.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            r = torch.from_numpy(r).float()
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_mean, s_sigma = torch.mean(s, dim=0), torch.std(s, dim=0)
            a_mean, a_sigma = torch.mean(a, dim=0), torch.std(a, dim=0)
            r_mean, r_sigma = torch.mean(r, dim=0), torch.std(r, dim=0)
            self.reward_net.set_transformations(s_mean, s_sigma, a_mean, a_sigma, r_mean, r_sigma)

        # get next state prediction
        sp = self.dynamics_net.forward(s, a).detach().clone()

        # call the generic fit function
        X = (s, a, sp) ; Y = r
        return fit_model(self.reward_net, X, Y, self.reward_opt, self.reward_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    def compute_path_rewards(self, paths):
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)
        if not self.learn_reward: 
            print("Reward model is not learned. Use the reward function from env.")
            return None
        s, a = paths['observations'], paths['actions']
        num_traj, horizon, s_dim = s.shape
        a_dim = a.shape[-1]
        s = s.reshape(-1, s_dim)
        a = a.reshape(-1, a_dim)
        r = self.reward(s, a)
        r = r.to('cpu').data.numpy().reshape(num_traj, horizon)
        paths['rewards'] = r


class DynamicsNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=(64,64),
                 s_mean = None,
                 s_sigma = None,
                 a_mean = None,
                 a_sigma = None,
                 out_mean = None,
                 out_sigma = None,
                 out_dim = None,
                 residual = True,
                 seed=123,
                 use_mask = True,
                 ):
        super(DynamicsNet, self).__init__()

        torch.manual_seed(seed)
        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.out_dim = state_dim if out_dim is None else out_dim
        self.layer_sizes = (state_dim + act_dim, ) + hidden_size + (self.out_dim, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.residual, self.use_mask = residual, use_mask
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma, out_mean, out_sigma)

    def set_transformations(self, s_mean=None, s_sigma=None,
                            a_mean=None, a_sigma=None,
                            out_mean=None, out_sigma=None):

        if s_mean is None:
            self.s_mean     = torch.zeros(self.state_dim)
            self.s_sigma    = torch.ones(self.state_dim)
            self.a_mean     = torch.zeros(self.act_dim)
            self.a_sigma    = torch.ones(self.act_dim)
            self.out_mean   = torch.zeros(self.out_dim)
            self.out_sigma  = torch.ones(self.out_dim)
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

        device = next(self.parameters()).data.device
        self.s_mean, self.s_sigma = self.s_mean.to(device), self.s_sigma.to(device)
        self.a_mean, self.a_sigma = self.a_mean.to(device), self.a_sigma.to(device)
        self.out_mean, self.out_sigma = self.out_mean.to(device), self.out_sigma.to(device)
        # if some state dimensions have very small variations, we will force it to zero
        self.mask = self.out_sigma >= 1e-6

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
        out = out * self.out_sigma + self.out_mean
        out = out * self.mask.float() if self.use_mask else out
        out = out + s if self.residual else out
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


class RewardNet(nn.Module):
    def __init__(self, state_dim, act_dim, 
                 hidden_size=(64,64),
                 s_mean = None,
                 s_sigma = None,
                 a_mean = None,
                 a_sigma = None,
                 seed=123,
                 ):
        super(RewardNet, self).__init__()
        torch.manual_seed(seed)
        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.layer_sizes = (state_dim + act_dim + state_dim, ) + hidden_size + (1, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma)

    def set_transformations(self, s_mean=None, s_sigma=None,
                            a_mean=None, a_sigma=None,
                            out_mean=None, out_sigma=None):

        if s_mean is None:
            self.s_mean, self.s_sigma       = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.a_mean, self.a_sigma       = torch.zeros(self.act_dim), torch.ones(self.act_dim)
            self.sp_mean, self.sp_sigma     = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.out_mean, self.out_sigma   = 0.0, 1.0 
        elif type(s_mean) == torch.Tensor:
            self.s_mean, self.s_sigma       = s_mean, s_sigma
            self.a_mean, self.a_sigma       = a_mean, a_sigma
            self.sp_mean, self.sp_sigma     = s_mean, s_sigma
            self.out_mean, self.out_sigma   = out_mean, out_sigma
        elif type(s_mean) == np.ndarray:
            self.s_mean, self.s_sigma       = torch.from_numpy(s_mean).float(), torch.from_numpy(s_sigma).float()
            self.a_mean, self.a_sigma       = torch.from_numpy(a_mean).float(), torch.from_numpy(a_sigma).float()
            self.sp_mean, self.sp_sigma     = torch.from_numpy(s_mean).float(), torch.from_numpy(s_sigma).float()
            self.out_mean, self.out_sigma   = out_mean, out_sigma
        else:
            print("Unknown type for transformations")
            quit()

        device = next(self.parameters()).data.device
        self.s_mean, self.s_sigma   = self.s_mean.to(device), self.s_sigma.to(device)
        self.a_mean, self.a_sigma   = self.a_mean.to(device), self.a_sigma.to(device)
        self.sp_mean, self.sp_sigma = self.sp_mean.to(device), self.sp_sigma.to(device)

        self.transformations = dict(s_mean=self.s_mean, s_sigma=self.s_sigma,
                                    a_mean=self.a_mean, a_sigma=self.a_sigma,
                                    out_mean=self.out_mean, out_sigma=self.out_sigma)

    def forward(self, s, a, sp):
        # The reward will be parameterized as r = f_theta(s, a, s').
        # If sp is unavailable, we can re-use s as sp, i.e. sp \approx s
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize all the inputs
        s = (s - self.s_mean) / (self.s_sigma + 1e-6)
        a = (a - self.a_mean) / (self.a_sigma + 1e-6)
        sp = (sp - self.sp_mean) / (self.sp_sigma + 1e-6)
        out = torch.cat([s, a, sp], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        out = out * (self.out_sigma + 1e-6) + self.out_mean
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = (self.s_mean, self.s_sigma,
                      self.a_mean, self.a_sigma)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_mean, s_sigma, a_mean, a_sigma = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_mean, s_sigma, a_mean, a_sigma)


def fit_model(nn_model, X, Y, optimizer, loss_func,
              batch_size, epochs, max_steps=1e10):
    """
    :param nn_model:        pytorch model of form Y = f(*X) (class)
    :param X:               tuple of necessary inputs to the function
    :param Y:               desired output from the function (tensor)
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :return:
    """

    assert type(X) == tuple
    for d in X: assert type(d) == torch.Tensor
    assert type(Y) == torch.Tensor
    device = Y.device
    for d in X: assert d.device == device

    num_samples = Y.shape[0]
    epoch_losses = []
    steps_so_far = 0
    for ep in tqdm(range(epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = 0.0
        num_steps = int(num_samples // batch_size)
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat    = nn_model.forward(*batch_X)
            loss = loss_func(Y_hat, batch_Y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.to('cpu').data.numpy()
        epoch_losses.append(ep_loss * 1.0/num_steps)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early..")
            break
    return epoch_losses
