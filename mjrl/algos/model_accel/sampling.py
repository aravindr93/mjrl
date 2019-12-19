import logging
import numpy as np
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
import torch
logging.disable(logging.CRITICAL)


# ===========================================================
# Rollout parameteric policy on fitted env to collect data
# ===========================================================

def policy_rollout(
        num_traj,
        env,
        policy,
        fitted_model,
        init_state=None,
        eval_mode=False,
        horizon=1e6,
        env_kwargs=None,
        seed=None,
):

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    if seed is not None:
        env.set_seed(seed)
        torch.manual_seed(seed)

    # get initial states
    if init_state is None:
        st = np.array([env.reset() for _ in range(num_traj)])
        st = torch.from_numpy(st).float()
    elif type(init_state) == np.ndarray:
        st = torch.from_numpy(init_state).float()
    elif type(init_state) == list:
        st = torch.from_numpy(np.array(init_state)).float()
    else:
        print("Unsupported format for init state")
        quit()

    # perform batched rollouts
    horizon = min(horizon, env.horizon)
    obs = []
    act = []
    for t in range(horizon):
        at = policy.model.forward(st)
        if eval_mode is not True:
            at = at + torch.randn(at.shape) * torch.exp(policy.log_std)
        stp1 = fitted_model.forward(st, at)
        obs.append(st.to('cpu').data.numpy())
        act.append(at.to('cpu').data.numpy())
        st = stp1

    obs = np.array(obs)
    obs = np.swapaxes(obs, 0, 1)  # (num_traj, horizon, state_dim)
    act = np.array(act)
    act = np.swapaxes(act, 0, 1)  # (num_traj, horizon, action_dim)
    paths = dict(observations = obs,
                 actions = act)

    return paths


# ===========================================================
# Rollout action sequences on the fitted model
# ===========================================================

def trajectory_rollout(actions, fitted_model, init_states):
    # init_states: (num_traj, state_dim) : numpy array
    # actions : (num_traj, horizon, action_dim) : numpy array
    # fitted_model : model(s, a) = s_tp1

    actions = np.array(actions) if type(actions) == list else actions
    num_traj = actions.shape[0]
    horizon = actions.shape[1]

    if len(init_states.shape) == 1:
        init_states = np.tile(init_states, (num_traj, 1))

    obs = []
    st = torch.from_numpy(init_states).float()
    for t in range(horizon):
        at = actions[:, t, :]
        at = torch.from_numpy(at).float()
        stp1 = fitted_model.forward(st, at)
        obs.append(st.data.numpy().copy())
        st = stp1

    obs = np.array(obs)
    obs = np.swapaxes(obs, 0, 1)
    paths = dict(observations=obs, actions=actions)
    return paths


# ===========================================================
# Rollout policy (parametric or implicit MPC) on real env
# ===========================================================
# TODO(Aravind): Remove redundancy. This can be coupled with the standard sample_paths in MJRL utils

def sample_paths(num_traj,
                 env,
                 policy,  # mpc policy on fitted model
                 horizon=1e6,
                 eval_mode=True,
                 base_seed=None,
                 noise_level=0.1,
                 ):

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env()
    else:
        print("Unsupported environment format")
        raise AttributeError
    if base_seed is not None:
        env.set_seed(base_seed)
    horizon = min(horizon, env.horizon)
    paths = []
    for ep in range(num_traj):
        env.reset()
        observations=[]
        actions=[]
        rewards=[]
        env_infos=[]
        t = 0
        done = False
        while t < horizon and done is False:
            obs = env.get_obs()
            ifo = env.get_env_infos()
            act = policy.get_action(obs)
            if eval_mode is False and type(act) != list:
                act = act + np.random.uniform(low=-noise_level, high=noise_level, size=act.shape[0])
            if type(act) == list:
                act = act[0] if eval_mode is False else act[1]['evaluation']
            next_obs, reward, done, _ = env.step(act)
            t = t + 1
            observations.append(obs)
            actions.append(act)
            rewards.append(reward)
            env_infos.append(ifo)
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminated=done,
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos)
        )
        paths.append(path)
    return paths


# ===========================================================
# Utility functions
# ===========================================================

def discount_sum(x, gamma, discounted_terminal=0.0):
    """
    discount sum a sequence with terminal value
    """
    y = []
    run_sum = discounted_terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])


def generate_perturbed_actions(base_act, filter_coefs):
    """
    Generate perturbed actions around a base action sequence
    """
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
    eps = base_act + eps
    eps[0] = eps[0] * (beta_0 + beta_1 + beta_2)
    eps[1] = beta_0 * eps[1] + (beta_1 + beta_2) * eps[0]
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
    return eps


def generate_paths(num_traj, fitted_model, start_state, base_act, filter_coefs, base_seed=None):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    if base_seed is not None:
        np.random.seed(base_seed)
    act_list = []
    for i in range(num_traj):
        act = generate_perturbed_actions(base_act, filter_coefs)
        act_list.append(act)
    act = np.array(act_list)
    paths = trajectory_rollout(act, fitted_model, start_state)
    return paths


def evaluate_policy(e, policy, fitted_model, noise_level=0.0,
                    real_step=False, num_episodes=10, visualize=False):
    # rollout the policy on env and record performance
    paths = []
    for ep in range(num_episodes):
        e.reset()
        observations = []
        actions = []
        rewards = []
        env_infos = []
        t = 0
        done = False
        while t < e.horizon and done is False:
            o = e.get_obs()
            ifo = e.get_env_infos()
            a = policy.get_action(o)
            if type(a) == list:
                a = a[1]['evaluation']
            if noise_level > 0.0:
                a = a + e.env.env.np_random.uniform(low=-noise_level, high=noise_level, size=a.shape[0])
            if real_step is False:
                next_s = fitted_model.predict(o, a)
                r = 0.0 # temporarily
                e.env.env.set_fitted_state(next_s)
            else:
                next_o, r, done, ifo2 = e.step(a)
                ifo = ifo2 if ifo == {} else ifo
            if visualize:
                e.render()

            t = t + 1
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            env_infos.append(ifo)

        path = dict(observations=np.array(observations), actions=np.array(actions),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos))
        if real_step is False:
            e.env.env.compute_path_rewards(path)
            try:
                path = e.env.env.truncate_paths([path])[0]
            except:
                pass
        paths.append(path)
        if visualize:
            print("episode score = %f " % np.sum(path['rewards']))
    return paths