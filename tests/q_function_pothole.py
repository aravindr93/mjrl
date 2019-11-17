
import pickle
import mjrl.envs

import numpy as np

from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QPi
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
from mjrl.utils.process_samples import compute_returns

import matplotlib.pyplot as plt

from mjrl.utils.evaluate_q_function import evaluate_n_step, evaluate_start_end, mse

policy_dir = 'pothole_point_mass_exp1/iterations/policy_10.pickle'
# policy_dir = 'pothole_point_mass_exp1/iterations/best_policy.pickle'
env_name = 'mjrl_pothole_point_mass-v0'

# policy_dir = 'point_mass_exp1/iterations/best_policy.pickle'
# env_name = 'mjrl_point_mass-v0'

K_train = 1000
K_test = 100
e = GymEnv(env_name)
time_dim = 3
hidden_size = (256, 256)
fit_lr = 1e-3
gamma = 0.96
batch_size = 64

num_fit_iters = 25
num_evals = 2
base_seed = 2

policy = pickle.load(open(policy_dir, 'rb'))
train_paths = sample_paths(K_train, e, policy, eval_mode=False, base_seed=base_seed)
test_paths = sample_paths(K_test, e, policy, eval_mode=True, base_seed=base_seed+1)
compute_returns(train_paths, gamma)
compute_returns(test_paths, gamma)
replay_buffer = TrajectoryReplayBuffer()
replay_buffer.push_many(train_paths)

# q_function = QPi(policy, e.observation_dim, e.action_dim, time_dim, e.horizon, replay_buffer,
#                     hidden_size=hidden_size, fit_lr=fit_lr, gamma=gamma, batch_size=batch_size, num_fit_iters=num_fit_iters)


q_function = QPi(policy, e.observation_dim, e.action_dim, 3, e.horizon, replay_buffer,
                batch_size=512, gamma=gamma, device='cuda',
                num_bellman_iters=30, num_fit_iters=300, fit_lr=1e-3,
                use_mu_approx=False, num_value_actions=5)


def evaluate(q_function, paths):
    # states = q_function.buffer['observations']
    # actions = q_function.buffer['actions']
    # times = q_function.buffer['time']
    # preds = q_function.forward(states, actions, times)
    total_error = 0.0
    n = 0
    for path in paths:
        n += path['observations'].shape[0]
        states = path['observations']
        actions = path['actions']
        times = path['time']
        Qs = path['returns']
        Qhats = q_function.predict(states, actions, times).reshape(-1)
        total_error += np.sum((Qs - Qhats)**2)
    return total_error / n

def plot_vs(q_function, t=0, title="figure"):
    xy_grid = np.mgrid[-1.4:1.4:0.05, -1.4:1.4:0.05]
    y, x = xy_grid
    xs_resh = x.reshape(-1)
    ys_resh = y.reshape(-1)
    xy = np.stack([xs_resh, ys_resh], axis=1)
    sqrtn = xy_grid.shape[1]
    # xy = xy_grid.reshape(-1, 2)
    n = xy.shape[0]
    states = np.concatenate([xy, np.zeros((n, 2)), np.ones((n, 2))], axis=1)
    times = np.zeros(n) + t
    values = q_function.compute_average_value(states, times)
    # return xy, values
    fig, ax = plt.subplots()
    ax.set_title(title)
    c = ax.pcolormesh(x, y, values[2].cpu().detach().numpy().reshape(sqrtn,sqrtn))
    fig.colorbar(c, ax=ax)

def plot_v_path(paths, t=0, title="figure"):
    n = len(paths)
    xs = np.empty(n)
    ys = np.empty(n)
    zs = np.empty(n)
    for i, path in enumerate(paths):
        x, y = path['observations'][t][0:2]
        ret = path['returns'][t]
        xs[i] = x
        ys[i] = y
        zs[i] = ret

    fig, ax = plt.subplots()
    ax.set_title(title)
    scatter = ax.scatter(xs, ys, c=zs)
    legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Returns t={}".format(t))
    ax.add_artist(legend)



all_losses = []
train_eval_scores = []
test_eval_scores = []
for eval_num in range(num_evals):
    losses, _ = q_function.bellman_update()
    train_eval_score = evaluate(q_function, train_paths)
    test_eval_score = evaluate(q_function, test_paths)
    train_eval_scores.append(train_eval_score)
    test_eval_scores.append(test_eval_score)
    all_losses += losses

    # if eval_num % 25 == 0:
    print(eval_num)

print('best train', np.min(train_eval_scores))
print('best test', np.min(test_eval_scores))

train_pred_1, train_mc_1 = evaluate_n_step(1, gamma, train_paths, q_function)
test_pred_1, test_mc_1 = evaluate_n_step(1, gamma, test_paths, q_function)
train_pred_end, train_mc_end = evaluate_start_end(gamma, train_paths, q_function)
test_pred_end, test_mc_end = evaluate_start_end(gamma, test_paths, q_function)

print('Train 1 step MSE', mse(train_pred_1, train_mc_1))
print('Test 1 step MSE', mse(test_pred_1, test_mc_1))
print('Train end step MSE', mse(train_pred_end, train_mc_end))
print('Test end step MSE', mse(test_pred_end, test_mc_end))

plt.figure('bellman losses')
plt.plot(all_losses)
plt.figure('train eval scores')
plt.plot(train_eval_scores)
plt.figure('test eval scores')
plt.plot(test_eval_scores)

plot_vs(q_function, title='Q t=0')
plot_vs(q_function, t=e.horizon, title='Q t=H')

plot_v_path(train_paths, t=0, title='Train Returns t=0')
plot_v_path(train_paths, t=e.horizon - 1, title='Train Returns t=H')

plot_v_path(test_paths, t=0, title='Test Returns t=0')
plot_v_path(test_paths, t=e.horizon-1, title='Test Returns t=H')

plt.show()
