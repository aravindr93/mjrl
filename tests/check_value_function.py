

# generate data

# learn Q function

# learn Value function

# compare the approximated values vs learned values



from mjrl.samplers.core import sample_paths
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QPi
from mjrl.utils.gym_env import GymEnv

from mjrl.baselines.shared_features_baseline import SharedFeaturesBaseline
from mjrl.utils.TupleMLP import TupleMLP
from mjrl.utils.process_samples import compute_returns
from mjrl.algos.model_accel.nn_dynamics import DynamicsModel
from mjrl.baselines.mlp_baseline import MLPBaseline

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
    FIT_ITERS = 5000
    BELLMAN_FIT_ITERS = 500
    BELLMAN_ITERS = 20

    q_baseline = SharedFeaturesBaseline(state_dim, action_dim, time_dim,
        replay_buffers, eval_policy, e.horizon,
        feature_size=FEAT_SIZE, hidden_sizes=hidden_sizes, shared_lr=SHARED_LR,
        both_lr=BOTH_LR, both_bellman_lr=BOTH_BELLMAN_LR,
        target_minibatch_size=TARGET_MB_SIZE, num_fit_iters=FIT_ITERS, gamma=args.gamma,
        device=args.device, bellman_minibatch_size=BELLMAN_MB, num_bellman_fit_iters=BELLMAN_FIT_ITERS,
        num_bellman_iters=BELLMAN_ITERS)



    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=100, learn_rate=1e-3)

    print('updating q_baseline')
    update_stats=q_baseline.update_returns()
    q_baseline.set_weight_newest()

    print('updating baseline')
    error_before, error_after=baseline.fit(paths[0], return_errors=True)
    
    compute_returns(eval_paths, args.gamma)


    def predict_and_plot(path, baseline, q_baseline):
        obs = torch.from_numpy(path['observations']).float().to(args.device)
        time = torch.from_numpy(path['time']).float().to(args.device)
        values_q_mu = q_baseline.value(obs, time)
        values_q_1 = q_baseline.value(obs, time, use_mu_approx=False, n_value=1)
        values_q_2 = q_baseline.value(obs, time, use_mu_approx=False, n_value=2)
        values_q_4 = q_baseline.value(obs, time, use_mu_approx=False, n_value=4)
        values_q_8 = q_baseline.value(obs, time, use_mu_approx=False, n_value=8)
        values = baseline.predict(path)

        vhqmu = values_q_mu.cpu().detach().squeeze().numpy()
        vhq1 = values_q_1.cpu().detach().squeeze().numpy()
        vhq2 = values_q_2.cpu().detach().squeeze().numpy()
        vhq4 = values_q_4.cpu().detach().squeeze().numpy()
        vhq8 = values_q_8.cpu().detach().squeeze().numpy()
        returns = path['returns']

        line_vhqmu, = plt.plot(vhqmu, label='Est values from Q (mu)')
        line_vhq1, = plt.plot(vhq1, label='Est values from Q (1)')
        line_vhq2, = plt.plot(vhq2, label='Est values from Q (2)')
        line_vhq4, = plt.plot(vhq4, label='Est values from Q (4)')
        line_vhq8, = plt.plot(vhq8, label='Est values from Q (8)')
        line_vh, = plt.plot(values, label='Est values from V')
        line_ret, = plt.plot(returns, label='discounted returns')
        plt.legend([line_vhqmu, line_vh, line_ret, line_vhq1, line_vhq2, line_vhq4, line_vhq8])
        plt.show()



