

from mjrl.utils.evaluate_policy import train_and_evaluate
from mjrl.samplers.core import sample_paths
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QPi
from mjrl.utils.gym_env import GymEnv


import argparse
import pickle
import mjrl.envs
import robel
import mj_envs

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # take in two policies, one to generate dataset, one to evaluate using that data

    parser = argparse.ArgumentParser(description='Evaluate a policy given data from a second policy')
    parser.add_argument('eval_policy_dir', type=str)
    parser.add_argument('data_policy_dir', type=str)
    parser.add_argument('env_name', type=str)
    parser.add_argument('--num_train_trajs', type=int, default=500)
    parser.add_argument('--num_eval_trajs', type=int, default=100)
    parser.add_argument('--eval_mode', type=bool, default=False)
    parser.add_argument('--plot_losses', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0xDEADBEEF)


    args = parser.parse_args()

    eval_policy = pickle.load(open(args.eval_policy_dir, 'rb'))
    data_policy = pickle.load(open(args.data_policy_dir, 'rb'))

    paths = sample_paths(args.num_train_trajs, args.env_name, data_policy, eval_mode=args.eval_mode, base_seed=args.seed)
    eval_paths = sample_paths(args.num_eval_trajs, args.env_name, eval_policy, eval_mode=True, base_seed=args.seed+1)
    
    # eval_policy.to(args.device)

    replay_buffer = TrajectoryReplayBuffer(device=args.device)
    replay_buffer.push_many(paths)
    e = GymEnv(args.env_name)

    init_q_function = None

    HIDDEN = 256
    # recon_weight = 0.0
    # reward_weight = 0.0
    recon_weight = 1e-1
    reward_weight = 1e-1

    # _, init_q_function = train_and_evaluate(eval_policy, replay_buffer, e, args.gamma, eval_paths, use_aux=True, init_function=None,
    #     num_traj=args.num_eval_trajs, print_info=True, num_bellman_iters=30, num_fit_iters=300,
    #     batch_size=4096, use_mu_approx=True, num_value_actions=15, recon_weight=recon_weight, reward_weight=reward_weight,
    #     reconstruction_hidden_sizes=(HIDDEN, HIDDEN), q_function_hidden_sizes=(HIDDEN,HIDDEN), device=args.device, base_seed=args.seed+2)

    eval_dict, q_function = train_and_evaluate(eval_policy, replay_buffer, e, args.gamma, eval_paths, use_aux=True, init_function=init_q_function,
        num_traj=args.num_eval_trajs, print_info=True, num_bellman_iters=10, num_fit_iters=300,
        batch_size=4096, use_mu_approx=True, num_value_actions=15, recon_weight=recon_weight, reward_weight=reward_weight,
        reconstruction_hidden_sizes=(HIDDEN, HIDDEN), q_function_hidden_sizes=(HIDDEN,HIDDEN), device=args.device, base_seed=args.seed+2)

    if args.plot_losses:
        fig, axes = plt.subplots(2, 4)
        axes[0, 0].plot(eval_dict['total_losses'])
        axes[0, 1].plot(eval_dict['bellman_losses'])
        axes[1, 0].plot(eval_dict['reconstruction_losses'])
        axes[1, 1].plot(eval_dict['reward_losses'])
        axes[0, 2].plot(eval_dict['eval_mse_1'])
        axes[1, 2].plot(eval_dict['eval_mse_end'])
        axes[0, 3].plot(eval_dict['eval_mse_1_true'])
        axes[1, 3].plot(eval_dict['eval_mse_end_true'])

        axes[0, 0].set_title('Total Loss')
        axes[0, 1].set_title('Bellman Loss')
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 1].set_title('Reward Loss')
        axes[0, 2].set_title('Q MSE 1 step')
        axes[1, 2].set_title('Q MSE end step')
        axes[0, 3].set_title('Q MSE 1 step (true gamma)')
        axes[1, 3].set_title('Q MSE end step (true gamma)')

        plt.show()




