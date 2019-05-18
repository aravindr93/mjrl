import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DATA_SMALLER = 1e-2
SIZE_SINGLE = 4608

class CNN:
    def __init__(self, action_dim,
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None,
                 robot_info_dim=None,
                 action_stats=None,
                 robot_info_stats=None,
                 use_late_fusion=True,
                 use_cuda=True):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.m = action_dim  # number of actions
        self.min_log_std = min_log_std
        self.use_cuda = use_cuda

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        out_shift = None
        out_scale = None
        robot_info_shift = None
        robot_info_scale = None
        if action_stats is not None:
            out_shift = action_stats['mean']
            out_scale = action_stats['std']
        if robot_info_stats is not None:
            robot_info_shift = robot_info_stats['mean']
            robot_info_scale = robot_info_stats['std']
        self.model = LeNet(self.m, robot_info_dim=robot_info_dim,
                           out_shift=out_shift, out_scale=out_scale, robot_info_shift=robot_info_shift,
                           robot_info_scale=robot_info_scale, use_late_fusion=use_late_fusion)

        if self.use_cuda:
            self.model.cuda()
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
            param.data = DATA_SMALLER * param.data
        if self.use_cuda:
            self.log_std = Variable((torch.ones(self.m) * init_log_std).cuda(), requires_grad=True)
        else:
            self.log_std = Variable((torch.ones(self.m) * init_log_std), requires_grad=True)

        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # ------------------------
        self.old_model = LeNet(self.m, robot_info_dim=robot_info_dim,
                               out_shift=out_shift, out_scale=out_scale, robot_info_shift=robot_info_shift,
                               robot_info_scale=robot_info_scale, use_late_fusion=use_late_fusion)
        if self.use_cuda:
            self.old_model.cuda()
            self.old_log_std = Variable((torch.ones(self.m) * init_log_std).cuda())
        else:
            self.old_log_std = Variable((torch.ones(self.m) * init_log_std))

        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.cpu().data.numpy().ravel())
        self.param_shapes = [p.cpu().data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.cpu().data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = None
        self.robot_info = None

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).cpu().data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.cpu().data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation, use_cuda=False, robot_info=None):
        o = observation.copy()
        o = np.expand_dims(o, axis=0)
        if robot_info is not None:
            robot_info = np.expand_dims(robot_info, 0)
        prem_shape = (0, 1, 4, 2, 3)
        try:
            self.obs_var.data = torch.from_numpy(o).permute(prem_shape).float()
            if robot_info is not None:
                self.robot_info = torch.from_numpy(robot_info).float()
            if use_cuda:
                self.obs_var.data = torch.from_numpy(o).permute(prem_shape).float().cuda()
                if robot_info is not None:
                    self.robot_info.data = torch.from_numpy(robot_info).float().cuda()
        except:
            self.obs_var = Variable(torch.from_numpy(o).float(), requires_grad=False)
            if robot_info is not None:
                self.robot_info = Variable(torch.from_numpy(robot_info).float(), requires_grad=False)
            if use_cuda:
                self.obs_var = Variable(torch.from_numpy(o).float().cuda(), requires_grad=False)
                if robot_info is not None:
                    self.robot_info = Variable(torch.from_numpy(robot_info).float().cuda(), requires_grad=False)
            self.obs_var = self.obs_var.permute(prem_shape)

        mean = self.model.forward(self.obs_var, robot_info=self.robot_info).cpu().data.numpy().ravel()
        return [mean, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1)

    def mean_LL(self, img_pixels, actions, model=None, log_std=None, robot_info=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        img_pixels_var = Variable(img_pixels.float(), requires_grad=False)
        if robot_info is not None:
            robot_info = robot_info.float()
        act_var = Variable(actions.float(), requires_grad=False)
        mean = model(img_pixels_var, robot_info=robot_info)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.cpu().data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions, robot_info=None):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std, robot_info=robot_info)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)

    def cpu(self):
        self.model.cpu()
        self.old_model.cpu()
        self.log_std.data = self.log_std.data.cpu()
        self.old_log_std.data = self.old_log_std.data.cpu()
        if self.obs_var is not None:
            self.obs_var.data = self.obs_var.data.cpu()
        if self.robot_info is not None:
            self.robot_info.data = self.robot_info.data.cpu()
        self.trainable_params = list(self.model.parameters()) + [self.log_std]
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

    def cuda(self):
        self.model.cuda()
        self.old_model.cuda()
        self.log_std.data = self.log_std.data.cuda()
        self.old_log_std.data = self.old_log_std.data.cuda()
        if self.obs_var is not None:
            self.obs_var.data = self.obs_var.data.cuda()
        if self.robot_info is not None:
            self.robot_info.data = self.robot_info.data.cuda()
        self.trainable_params = list(self.model.parameters()) + [self.log_std]
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()


class LeNet(nn.Module):
    def __init__(self, act_dim, robot_info_dim=None,
                 out_shift=None, out_scale=None, robot_info_shift=None, robot_info_scale=None, use_late_fusion=True):
        super(LeNet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        if use_late_fusion:
            print('Using Late Fusion')
            self.conv1 = nn.Conv2d(3, 16, 3)
        else:
            print('Using Early Fusion')
            self.conv1 = nn.Conv2d(9, 16, 3)

        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv4_bn = nn.BatchNorm2d(32)

        self.use_late_fusion = torch.ByteTensor([use_late_fusion])
        self.set_transformations(out_shift, out_scale, robot_info_shift, robot_info_scale, act_dim)

        size = SIZE_SINGLE * 3 if self.use_late_fusion.all() else SIZE_SINGLE
        self.fc1 = nn.Linear(size, 200)

        if robot_info_dim is not None:
            self.fc2 = nn.Linear(200 + robot_info_dim, 128 + robot_info_dim)
            self.fc3 = nn.Linear(128 + robot_info_dim, act_dim)
        else:
            self.fc2 = nn.Linear(200, act_dim)

    def forward(self, x, robot_info):
        x_size = x.size()
        x /= 255.0
        x *= 2
        x -= 1
        if self.use_late_fusion.all():
            x = x.contiguous().view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        else:
            x = x.contiguous().view(x_size[0], x_size[1] * x_size[2], x_size[3], x_size[4])

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2, stride=2)
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2, stride=2)
        out = self.dropout(out)
        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2, stride=2)
        out = F.relu(self.conv4_bn(self.conv4(out)))

        if self.use_late_fusion.all():
            out = out.view(x_size[0], x_size[1], out.size(1), out.size(2), out.size(3))
        out = out.view(out.size(0), -1)

        out = F.tanh(self.fc1(out))
        if robot_info is not None:
            if self.robot_info_shift is not None and self.robot_info_scale is not None:
                robot_info = (robot_info - self.robot_info_shift) / (self.robot_info_scale + 1e-8)
            out = torch.cat((out, robot_info), 1)
            out = F.tanh(self.fc2(out))
            out = self.fc3(out)
            if self.out_shift is not None and self.out_scale is not None:
                out = out * self.out_scale + self.out_shift
            return out

        out = self.fc2(out)

        if self.out_scale is not None and self.out_shift is not None:
            out = out * self.out_scale + self.out_shift

        return out

    def set_transformations(self, out_shift, out_scale, robot_info_shift, robot_info_scale, action_dim):
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(action_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(action_dim)

        self.robot_info_shift = torch.from_numpy(
            np.float32(robot_info_shift)) if robot_info_shift is not None else torch.zeros(action_dim)
        self.robot_info_scale = torch.from_numpy(
            np.float32(robot_info_scale)) if robot_info_scale is not None else torch.ones(action_dim)

    def cuda(self):
        super(LeNet, self).cuda()

        self.use_late_fusion = self.use_late_fusion.cuda()
        if self.robot_info_shift is not None and self.robot_info_scale is not None:
            self.robot_info_shift = self.robot_info_shift.cuda()
            self.robot_info_scale = self.robot_info_scale.cuda()

        if self.out_scale is not None and self.out_shift is not None:
            self.out_shift = self.out_shift.cuda()
            self.out_scale = self.out_scale.cuda()

    def cpu(self):
        super(LeNet, self).cpu()

        self.use_late_fusion = self.use_late_fusion.cpu()
        if self.robot_info_shift is not None and self.robot_info_scale is not None:
            self.robot_info_shift = self.robot_info_shift.cpu()
            self.robot_info_scale = self.robot_info_scale.cpu()

        if self.out_scale is not None and self.out_shift is not None:
            self.out_shift = self.out_shift.cpu()
            self.out_scale = self.out_scale.cpu()
