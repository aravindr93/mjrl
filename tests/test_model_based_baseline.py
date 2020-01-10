

from mjrl.samplers.core import sample_paths
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QPi
from mjrl.utils.gym_env import GymEnv

from mjrl.baselines.model_based_baseline import MBBaselineNaive
from mjrl.baselines.model_based_baseline import MBBaselineDoubleV
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


if __name__ == '__main__':
    # take in two policies, one to generate dataset, one to evaluate using that data

    parser = argparse.ArgumentParser(description='Test model based baseline')
    parser.add_argument('data_policy_dir', type=str)
    parser.add_argument('eval_policy_dir', type=str)
    parser.add_argument('env_name', type=str)
    parser.add_argument('--num_train_trajs', type=int, default=500)
    parser.add_argument('--num_eval_trajs', type=int, default=100)
    parser.add_argument('--eval_mode', type=bool, default=False)
    parser.add_argument('--plot_losses', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0xDEADBEEF)


    args = parser.parse_args()

    data_policy = pickle.load(open(args.data_policy_dir, 'rb'))
    evals_policy = pickle.load(open(args.eval_policy_dir, 'rb'))

    paths = sample_paths(args.num_train_trajs, args.env_name, data_policy, eval_mode=args.eval_mode, base_seed=args.seed)
    eval_paths = sample_paths(args.num_eval_trajs, args.env_name, evals_policy, eval_mode=True, base_seed=args.seed+1)
    
    # eval_policy.to(args.device)

    replay_buffer = TrajectoryReplayBuffer(device=args.device)
    replay_buffer.push_many(paths)
    e = GymEnv(args.env_name)

    state_dim = e.observation_dim
    action_dim = e.action_dim

    HIDDEN_SIZE = 512
    VALUE_HIDDEN = 512
    time_dim = 3

    # dynamics_model = pickle.load(open(args.dynamics_dir, 'rb'))
    dynamics_model = DynamicsModel(state_dim=e.observation_dim, act_dim=e.action_dim,
            hidden_size=(HIDDEN_SIZE, HIDDEN_SIZE), fit_lr=2e-4) 
    # model_network = TupleMLP(state_dim + action_dim, state_dim, (HIDDEN_SIZE, HIDDEN_SIZE), non_linearity=torch.nn.Tanh)
    value_network = TupleMLP(state_dim + time_dim, 1, (VALUE_HIDDEN, VALUE_HIDDEN))
    target_value_network = TupleMLP(state_dim + time_dim, 1, (VALUE_HIDDEN, VALUE_HIDDEN))

    def reward_func(state, action):
        return e.env.get_reward(state, action)

    # baseline = MBBaselineNaive(dynamics_model, value_network, target_value_network,
    #     data_policy, reward_func, replay_buffer, args.horizon, e.horizon, args.gamma,
    #     model_lr=1e-2, model_batch_size=64, num_model_fit_iters=100,
    #     num_bellman_iters=10, num_bellman_fit_iters=250, bellman_batch_size=512)
    
    # update_stats = baseline.update()
    
    baseline = MBBaselineDoubleV(dynamics_model, value_network, target_value_network,
        data_policy, reward_func, replay_buffer, args.horizon, e.horizon, args.gamma,
        lr=1e-2, batch_size=64, num_bellman_iters=10, num_bellman_fit_iters=250,
        reward_weight=1.0, value_weight=1.0)

    baseline.update()

    compute_returns(eval_paths, args.gamma)

    def test(H, use_average=False, n_rollouts=4, print_info=True):
        total_model_mse = 0.0
        total_no_model_mse = 0.0

        baseline.H = H

        for path in eval_paths:
            obs = torch.from_numpy(path['observations']).float()
            time = torch.from_numpy(path['time']).float()
            returns = torch.from_numpy(path['returns']).view(-1, 1).float()
            values_model = baseline.value(obs, time, use_average=use_average, n_rollouts=n_rollouts)
            values_no_model = baseline.value_no_model(obs, time)
            model_mse = torch.nn.functional.mse_loss(returns, values_model)
            no_model_mse = torch.nn.functional.mse_loss(returns, values_no_model)
            total_model_mse += model_mse
            total_no_model_mse += no_model_mse
            # print('model', model_mse, 'no model', no_model_mse)
        if print_info:
            print('model total', total_model_mse, 'no model total', total_no_model_mse)
        return total_model_mse, total_no_model_mse

    def test_learned_vs_mujoco(H):
        pass

    def predict_and_plot(path, n_rollouts=2, add_tvf=True):
        obs = torch.from_numpy(path['observations']).float().to(args.device)
        time = torch.from_numpy(path['time']).float().to(args.device)
        values_hat = baseline.value(obs, time, n_rollouts=n_rollouts, add_tvf=add_tvf)
        values_hat_no_model = baseline.value_no_model(obs, time)

        vh = values_hat.cpu().detach().squeeze().numpy()
        vhnm = values_hat_no_model.cpu().detach().squeeze().numpy()
        returns = path['returns']

        line_vh, = plt.plot(vh, label='estimated values')
        line_vhnm, = plt.plot(vhnm, label='est values (no model)')
        line_ret, = plt.plot(returns, label='discounted returns')
        plt.legend([line_vh, line_vhnm, line_ret])
        plt.show()

    test(0)
    test(1)
    test(args.horizon)







