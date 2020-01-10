

from mjrl.samplers.core import sample_paths
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QPi
from mjrl.utils.gym_env import GymEnv

from mjrl.baselines.shared_features_baseline import SharedFeaturesBaseline
from mjrl.utils.TupleMLP import TupleMLP
from mjrl.utils.process_samples import compute_returns
from mjrl.algos.model_accel.nn_dynamics import DynamicsModel

import argparse
import torch
import pickle
import mjrl.envs
import robel
import mj_envs

import matplotlib.pyplot as plt
import numpy as np

from mjrl.utils.process_samples import compute_returns

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # take in two policies, one to generate dataset, one to evaluate using that data

    parser = argparse.ArgumentParser(description='Test shared features baseline')
    parser.add_argument('env_name', type=str)
    parser.add_argument('eval_policy_dir', type=str)
    parser.add_argument('data_policy_dir', type=str, nargs='*')
    parser.add_argument('--num_train_trajs', type=int, default=10)
    parser.add_argument('--num_eval_trajs', type=int, default=10)
    parser.add_argument('--eval_mode', type=bool, default=False)
    parser.add_argument('--plot_losses', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0xDEADBEEF)


    args = parser.parse_args()

    eval_policy = pickle.load(open(args.eval_policy_dir,'rb'))
    policies = [pickle.load(open(pol_dir, 'rb')) for pol_dir in args.data_policy_dir]

    paths = [sample_paths(args.num_train_trajs, args.env_name, pol, eval_mode=args.eval_mode, base_seed=args.seed) for pol in policies]
    eval_paths = sample_paths(args.num_eval_trajs, args.env_name, eval_policy, eval_mode=True, base_seed=args.seed+1)
    # eval_paths = pickle.load(open('/home/ben/builds/mjrl/mjrl/utils/paths.pickle', 'rb'))
    # eval_paths = [eval_paths[1], eval_paths[3]]
    # eval_paths = pickle.load(open('eval_paths.pickle', 'rb'))
    
    replay_buffers = []

    for path in paths:
        replay_buffer = TrajectoryReplayBuffer(device=args.device)
        compute_returns(path, args.gamma)
        replay_buffer.push_many(path)
        replay_buffers.append(replay_buffer)

    e = GymEnv(args.env_name)

    state_dim = e.observation_dim
    action_dim = e.action_dim

    HIDDEN_SIZE = 1024
    hidden_sizes=(HIDDEN_SIZE, HIDDEN_SIZE)
    FEAT_SIZE = 1024
    SHARED_LR = 5e-4
    BOTH_LR = 3e-6
    BOTH_BELLMAN_LR = 5e-6
    TARGET_MB_SIZE = 512
    BELLMAN_MB = 256
    time_dim = 0
    FIT_ITERS = 10000
    BELLMAN_FIT_ITERS = 500
    BELLMAN_ITERS = 20

    baseline = SharedFeaturesBaseline(state_dim, action_dim, time_dim,
        replay_buffers, eval_policy, e.horizon,
        feature_size=FEAT_SIZE, hidden_sizes=hidden_sizes, shared_lr=SHARED_LR,
        both_lr=BOTH_LR, both_bellman_lr=BOTH_BELLMAN_LR,
        target_minibatch_size=TARGET_MB_SIZE, num_fit_iters=FIT_ITERS, gamma=args.gamma,
        device=args.device, bellman_minibatch_size=BELLMAN_MB, num_bellman_fit_iters=BELLMAN_FIT_ITERS,
        num_bellman_iters=BELLMAN_ITERS)

    update_stats = baseline.update_returns()

    compute_returns(eval_paths, args.gamma)

    def predict_and_plot(path):
        obs = torch.from_numpy(path['observations']).float().to(args.device)
        time = torch.from_numpy(path['time']).float().to(args.device)
        values_hat = baseline.value(obs, time)

        vh = values_hat.cpu().detach().squeeze().numpy()
        returns = path['returns']

        line_vh, = plt.plot(vh, label='estimated values')
        line_ret, = plt.plot(returns, label='discounted returns')
        plt.legend([line_vh, line_ret])
        plt.show()


    def test(paths, i=None):
        total_mse = 0.0

        all_mses = []

        for path in paths:
            obs = torch.from_numpy(path['observations']).float().to(args.device)
            time = torch.from_numpy(path['time']).float().to(args.device)
            returns = torch.from_numpy(path['returns']).view(-1, 1).float().to(args.device)
            if i is None:
                values_hat = baseline.value(obs, time)
            else:
                action = policies[i].get_action_pytorch(obs).to(args.device)
                x = baseline.featurize_state_action_time(obs, action, time)
                shared_features = baseline.shared_features_network(x)
                values_hat = baseline.linear_q_weights[i](shared_features)
            model_mse = torch.nn.functional.mse_loss(returns, values_hat)
            all_mses.append(model_mse.item())
            total_mse += model_mse
        
        return total_mse, all_mses
    
    def test_linear(n_update):
        all_mses = []
        for _ in range(n_update):
            losses = baseline.fit_targets_linear()
            baseline.target_policy_linear.load_state_dict(baseline.policy_linear.state_dict())
            mse = test(eval_paths)
            all_mses.append(mse)
        return all_mses

    def test_loss(policy):
        lin_weight = baseline.linear_q_weights[policy]
        samples = baseline.replay_buffers[policy].get_sample(1000)
        y = samples['returns'].view(-1, 1)
        state = samples['observations']
        action = samples['actions']
        time = samples['time']
        x = baseline.featurize_state_action_time(state, action, time)
        shared_features = baseline.shared_features_network(x)
        y_hat = lin_weight(shared_features)

        mse = torch.nn.functional.mse_loss(y_hat, y)
        return mse

    def test_bellman_loss(policy):
        lin_weight = baseline.linear_q_weights[policy]
        samples = baseline.replay_buffers[policy].get_sample(1000)
        y = samples['returns'].view(-1, 1)
        state = samples['observations']
        next_state = samples['next_observations']
        action = samples['actions']
        rewards = samples['rewards'].view(-1,1)
        next_action = policies[policy].get_action_pytorch(next_state).to(args.device)
        time = samples['time']
        x = baseline.featurize_state_action_time(state, action, time)
        shared_features = baseline.shared_features_network(x)
        Q_s = lin_weight(shared_features)

        x = baseline.featurize_state_action_time(next_state, next_action, time+1)
        shared_features_sp = baseline.shared_features_network(x)
        Q_s_hat = rewards + args.gamma * lin_weight(shared_features_sp)
        bellman_error = torch.nn.functional.mse_loss(Q_s, Q_s_hat)

        return bellman_error

    def add_on_policy_data():
        eval_rb = TrajectoryReplayBuffer(device=args.device)
        eval_rb.push_many(eval_paths)
        policies.append(eval_policy)
        baseline.new_replay_buffer(eval_rb)
    # init_mse = test(eval_paths)
    # _ = baseline.update_bellman_only_linear()
    # only_linear = test(eval_paths)



