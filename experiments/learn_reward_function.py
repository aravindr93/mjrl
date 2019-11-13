
import gym
import mjrl.envs
from mjrl.policies.gaussian_mlp import MLP
from mjrl.samplers.core import sample_data_batch
from mjrl.utils.TupleMLP import TupleMLP
import numpy as np
from mjrl.utils.gym_env import GymEnv
import torch.nn.functional as F
import torch
import pickle

def paths_to_reward_dataset(paths, exclude_last=False):
    Xs = []
    Ys = []
    for path in paths:
        n = path['observations'].shape[0]
        Xs.append(np.concatenate([path['observations'][:n-exclude_last], path['actions'][:n-exclude_last]], axis=1))
        Ys.append(path['rewards'][:n-exclude_last])
    
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

def paths_to_dynamics_dataset(paths):
    Xs = []
    Ys = []
    for path in paths:
        obs = path['observations']
        n = obs.shape[0]
        Xs.append(np.concatenate([obs[:n-1], path['actions'][:n-1]], axis=1))
        Ys.append(obs[1:n])
    
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

def train(Xs, Ys, network, optim, batch_size, iters, print_info=False):
    N = Xs.shape[0]
    for iter in range(iters):
        optim.zero_grad()
        idx = np.random.choice(N, size=batch_size)
        Y_hat = network(Xs[idx])
        loss = F.mse_loss(Y_hat, Ys[idx])
        loss.backward()
        optim.step()

        if iter % 100 == 0 and print_info:
            print(loss.item())


def shuffle_split_torch_dataset(Xs, Ys, N_train, shuffle_idxs=None, ret_idx=False):
    N = Xs.shape[0]

    if shuffle_idxs is None:
        shuffle_idxs = np.random.choice(N, N, replace=False)
    
    Xs = Xs[shuffle_idxs]
    Ys = Ys[shuffle_idxs]

    train_Xs = torch.from_numpy(Xs[:N_train]).float()
    train_Ys = torch.from_numpy(Ys[:N_train]).float()

    test_Xs = torch.from_numpy(Xs[N_train:]).float()
    test_Ys = torch.from_numpy(Ys[N_train:]).float()

    if len(train_Ys.shape) == 1:
        train_Ys = train_Ys.unsqueeze(1)
        test_Ys = test_Ys.unsqueeze(1)
    
    if ret_idx:
        return train_Xs, train_Ys, test_Xs, test_Ys, shuffle_idxs
    return train_Xs, train_Ys, test_Xs, test_Ys

def train_dynamics(hidden_sizes=(256, 256), print_info=False,
    N=20000, N_train=16000, lr=1e-3,
    batch_size = 256, iters = 2000, num_cpu=1):
    e_s = GymEnv('HopperStateVel-v0')
    state_policy = pickle.load(open('npg_state_hopper/iterations/best_policy.pickle', 'rb'))
    
    state_paths = sample_data_batch(N, e_s, state_policy, num_cpu=num_cpu)

    state_Xs, state_Ys = paths_to_dynamics_dataset(state_paths)

    state_train_Xs, state_train_Ys, state_test_Xs, state_test_Ys \
        = shuffle_split_torch_dataset(state_Xs, state_Ys, N_train)

    d_in = e_s.action_space.shape[0] + e_s.observation_space.shape[0]
    d_out = e_s.observation_space.shape[0]

    state_dynamics_network = TupleMLP(d_in, d_out, hidden_sizes)

    state_optim = torch.optim.Adam(state_dynamics_network.parameters(), lr)
    train(state_train_Xs, state_train_Ys, state_dynamics_network, state_optim, batch_size, iters)

    state_Y_hat = state_dynamics_network(state_test_Xs)
    state_mse = F.mse_loss(state_Y_hat, state_test_Ys)

    if print_info:
        print('state_mse', state_mse)

    return state_dynamics_network


def reward(Xs):
    return 1 + Xs[:, 5:6]

def train_no_aux(hidden_sizes=(256, 256), print_info=False,
    N=20000, N_train=16000, lr=1e-3,
    batch_size = 256, iters = 2000, num_cpu=1):
    e_s = GymEnv('HopperStateVel-v0')
    e_nexts = GymEnv('HopperNextStateVel-v0')
    
    # behavioral_policy = MLP(e_s)
    state_policy = pickle.load(open('npg_state_hopper/iterations/best_policy.pickle', 'rb'))
    next_state_policy = pickle.load(open('npg_state_hopper/iterations/best_policy.pickle', 'rb'))

    state_paths = sample_data_batch(N, e_s, state_policy, num_cpu=num_cpu)
    next_state_paths = sample_data_batch(N, e_nexts, next_state_policy, num_cpu=num_cpu)

    state_Xs, state_Ys = paths_to_reward_dataset(state_paths)
    next_state_Xs, next_state_Ys = paths_to_reward_dataset(next_state_paths)

    state_train_Xs, state_train_Ys, state_test_Xs, state_test_Ys \
        = shuffle_split_torch_dataset(state_Xs, state_Ys, N_train)
    next_state_train_Xs, next_state_train_Ys, next_state_test_Xs, next_state_test_Ys \
        = shuffle_split_torch_dataset(next_state_Xs, next_state_Ys, N_train)

    d_in = e_s.action_space.shape[0] + e_s.observation_space.shape[0]

    state_reward_network = TupleMLP(d_in, 1, hidden_sizes)
    next_state_reward_network = TupleMLP(d_in, 1, hidden_sizes)

    state_optim = torch.optim.Adam(state_reward_network.parameters(), lr)
    next_state_optim = torch.optim.Adam(next_state_reward_network.parameters(), lr)

    train(state_train_Xs, state_train_Ys, state_reward_network, state_optim, batch_size, iters)
    train(next_state_train_Xs, next_state_train_Ys, next_state_reward_network, next_state_optim, batch_size, iters)

    state_Y_hat = state_reward_network(state_test_Xs)
    next_state_Y_hat = next_state_reward_network(next_state_test_Xs)

    state_mse = F.mse_loss(state_Y_hat, state_test_Ys)
    next_state_mse = F.mse_loss(next_state_Y_hat, next_state_test_Ys)

    if print_info:
        print('state_mse', state_mse, 'next_state_mse', next_state_mse)
    
    return state_reward_network, next_state_reward_network, state_test_Xs, state_test_Ys, next_state_test_Xs, next_state_test_Ys

def train_aux(hidden_sizes=(256, 256), print_info=False, recon_weight=1.0,
    reward_weight=1.0, N=20000, N_train=16000, lr=1e-3,
    batch_size = 256, iters = 2000, num_cpu=1):
    e_s = GymEnv('HopperStateVel-v0')
    
    state_policy = pickle.load(open('npg_state_hopper/iterations/best_policy.pickle', 'rb'))

    #######
    e_nexts = GymEnv('HopperNextStateVel-v0')
    state_paths = sample_data_batch(N, e_nexts, state_policy, num_cpu=num_cpu)
    #######

    # state_paths = sample_data_batch(N, e_s, state_policy, num_cpu=num_cpu)

    state_Xs, state_rewards = paths_to_reward_dataset(state_paths, exclude_last=True)
    # state_Xs, state_rewards = paths_to_reward_dataset(state_paths, exclude_last=True)

    _, next_states = paths_to_dynamics_dataset(state_paths)

    state_train_Xs, state_train_Ys, state_test_Xs, state_test_Ys, shuffle_idx \
        = shuffle_split_torch_dataset(state_Xs, state_rewards, N_train, ret_idx=True)

    _, state_train_next_states, _, state_test_next_states \
        = shuffle_split_torch_dataset(state_Xs, next_states, N_train, shuffle_idxs=shuffle_idx)

    d_state = e_s.observation_space.shape[0]
    d_in = e_s.action_space.shape[0] + d_state
    
    dynamics_network = TupleMLP(d_in, d_state, hidden_sizes)
    reward_network = TupleMLP(d_state, 1, hidden_sizes)

    optim = torch.optim.Adam(list(dynamics_network.parameters()) + list(reward_network.parameters()), lr)

    for iter in range(iters):
        optim.zero_grad()
        idx = np.random.choice(N_train, size=batch_size)
        next_state_hat = dynamics_network(state_train_Xs[idx])
        recon_loss = F.mse_loss(next_state_hat, state_train_next_states[idx])

        reward_hat = reward_network(next_state_hat)
        reward_loss = F.mse_loss(reward_hat, state_train_Ys[idx])

        loss = recon_weight * recon_loss + reward_weight * reward_loss

        loss.backward()
        optim.step()

        if iter % 100 == 0 and print_info:
            print(loss.item())

    next_state_hat = dynamics_network(state_test_Xs)
    reward_hat = reward_network(next_state_hat)

    recon_mse = F.mse_loss(next_state_hat, state_test_next_states)
    reward_mse = F.mse_loss(reward_hat, state_test_Ys)

    if print_info:
        print('recon_mse', recon_mse, 'reward_mse', reward_mse)

    return dynamics_network, reward_network

def test_all(hidden_size):
    hidden_size_single = (hidden_size, hidden_size)

    N = 100000
    N_train = 90000
    lr = 2e-4
    batch_size = 64
    iters = 10000
    num_cpu = 1

    dynamics_network = train_dynamics(hidden_size_single,
        N=N, N_train=N_train, lr=lr, batch_size=batch_size, iters=iters, num_cpu=num_cpu)

    state_reward_network, next_state_reward_network, \
    state_test_Xs, state_test_Ys, next_state_test_Xs, next_state_test_Ys \
        = train_no_aux(hidden_size_single, N=N, N_train=N_train,
            lr=lr, batch_size=batch_size, iters=iters, num_cpu=num_cpu)

    joint_dynamics_network, joint_reward_network = train_aux(hidden_size_single, reward_weight=10.0, 
        N=N, N_train=N_train, lr=lr, batch_size=batch_size, iters=iters, num_cpu=num_cpu)

    state_primes = dynamics_network(next_state_test_Xs)
    acts = torch.zeros(state_primes.shape[0], 3)
    sp_act = torch.cat((state_primes, acts),dim=1)

    Y_hat1 = reward(dynamics_network(next_state_test_Xs))
    Y_hat2 = state_reward_network(sp_act)
    Y_hat3 = next_state_reward_network(next_state_test_Xs)
    Y_hat4 = joint_reward_network(joint_dynamics_network(next_state_test_Xs))

    mse1 = F.mse_loss(Y_hat1, next_state_test_Ys)
    mse2 = F.mse_loss(Y_hat2, next_state_test_Ys)
    mse3 = F.mse_loss(Y_hat3, next_state_test_Ys)
    mse4 = F.mse_loss(Y_hat4, next_state_test_Ys)

    print('Learned Dynamics Known Reward mse', mse1.item())
    print('Learned Dynamics Learned Reward Separate', mse2.item())
    print('Learned Next Reward', mse3.item())
    print('Learned Dynamics Learned Reward Joint', mse4.item())

# if __name__ == '__main__':
    
    
