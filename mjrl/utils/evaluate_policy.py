
from mjrl.utils.evaluate_q_function import evaluate_n_step, evaluate_start_end, mse
from mjrl.utils.networks import QPi

import mjrl.samplers.core as trajectory_sampler


def train_and_evaluate(policy, replay_buffer, env, gamma, eval_paths, init_function=None, num_traj=100, print_info=False, use_aux = False, base_seed=123, **kwargs):
    """
    Given paths and policy, train and evaluate the q_function
    """

    q_function = QPi(policy, env.observation_dim, env.action_dim, 3, env.horizon, replay_buffer,
        gamma=gamma, use_auxilary=use_aux, **kwargs)
    
    if init_function:
        # q_function.network.set_params(init_function.network.get_params())
        # q_function.target_network.set_params(init_function.target_network.get_params())
        q_function.network.set_q_params(init_function.network.get_q_params())
        q_function.target_network.set_q_params(init_function.target_network.get_q_params())

    if use_aux:
        total_losses, bellman_losses, reconstruction_losses, reward_losses, \
            update_time, eval_mse_1, eval_mse_end, eval_mse_1_true, eval_mse_end_true = q_function.bellman_update(all_losses=use_aux, eval_paths=eval_paths)
    else:
        bellman_losses, update_time, eval_mse_1, eval_mse_end, eval_mse_1_true, eval_mse_end_true = q_function.bellman_update(all_losses=use_aux, eval_paths=eval_paths)
        reconstruction_losses = [0.0] * len(bellman_losses)
        reward_losses = [0.0] * len(bellman_losses)
        total_losses = bellman_losses
    # total_losses, bellman_losses, reconstruction_losses, reward_losses, \
    #     update_time = q_function.bellman_update(all_losses=True, eval_paths=None)

    input_dict = dict(num_traj=num_traj, env=env, policy=policy, horizon=env.horizon,
                                base_seed=base_seed, num_cpu=1)
    train_paths = trajectory_sampler.sample_paths(**input_dict)
    
    test_paths = eval_paths

    train_pred_1, train_mc_1 = evaluate_n_step(1, gamma, train_paths, q_function)
    test_pred_1, test_mc_1 = evaluate_n_step(1, gamma, test_paths, q_function)
    train_pred_end, train_mc_end = evaluate_start_end(gamma, train_paths, q_function)
    test_pred_end, test_mc_end = evaluate_start_end(gamma, test_paths, q_function)

    train_1_mse = mse(train_pred_1, train_mc_1)
    test_1_mse = mse(test_pred_1, test_mc_1)
    train_end_mse = mse(train_pred_end, train_mc_end)
    test_end_mse = mse(test_pred_end, test_mc_end)

    if print_info:
        print('Train 1 step MSE', train_1_mse)
        print('Test 1 step MSE', test_1_mse)
        print('Train end step MSE', train_end_mse)
        print('Test end step MSE', test_end_mse)
        print('time', update_time)

    ret_dict = {
        'total_losses': total_losses,
        'bellman_losses': bellman_losses,
        'reconstruction_losses': reconstruction_losses,
        'reward_losses': reward_losses,
        'update_time': update_time,
        'train_1_mse': train_1_mse,
        'test_1_mse': test_1_mse,
        'train_end_mse': train_end_mse,
        'test_end_mse': test_end_mse,
        'eval_mse_1': eval_mse_1,
        'eval_mse_end': eval_mse_end,
        'eval_mse_1_true': eval_mse_1_true,
        'eval_mse_end_true': eval_mse_end_true
    }

    return ret_dict, q_function




