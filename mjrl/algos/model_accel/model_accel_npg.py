import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import os
import time as timer
from torch.autograd import Variable
from mjrl.utils.gym_env import GymEnv
from mjrl.algos.model_accel.nn_dynamics import DynamicsModel

import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.algos.model_accel.sampling import policy_rollout

# Import NPG
from mjrl.algos.npg_cg import NPG


class ModelAccelNPG(NPG):
    def __init__(self, fitted_model=None,
                 refine=False,
                 kappa=5.0,
                 plan_horizon=10,
                 plan_paths=100,
                 **kwargs):
        super(ModelAccelNPG, self).__init__(**kwargs)
        if fitted_model is None:
            print("Algorithm requires a NN dynamics model (or list of fitted models)")
            quit()
        elif isinstance(fitted_model, DynamicsModel):
            self.fitted_model = [fitted_model]
        else:
            self.fitted_model = fitted_model
        self.refine = refine
        self.kappa, self.plan_horizon, self.plan_paths = kappa, plan_horizon, plan_paths

    def to(self, device):
        # Convert all the networks (except policy network which is clamped to CPU)
        # to the specified device
        for model in self.fitted_model:
            model.to(device)
        self.baseline.model.to(device)

    def is_cuda(self):
        # Check if any of the networks are on GPU
        model_cuda = [model.is_cuda() for model in self.fitted_model]
        model_cuda = any(model_cuda)
        baseline_cuda = next(self.baseline.model.parameters()).is_cuda
        return any([model_cuda, baseline_cuda])

    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   ):

        ts = timer.time()

        # get the correct env behavior
        if env is None:
            env = self.env
        elif type(env) == str:
            env = GymEnv(env)
        elif isinstance(env, GymEnv):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        # generate paths with fitted dynamics
        # we want to use the same task instances (e.g. goal locations) for each model in ensemble
        paths = []

        # NOTE: When running on hardware, we need to load the set of initial states from a pickle file
        # init_states = pickle.load(open(<some_file>.pickle, 'rb'))
        # init_states = init_states[:N]
        init_states = np.array([env.reset() for _ in range(N)])

        for model in self.fitted_model:
            # dont set seed explicitly -- this will make rollouts follow tne global seed
            rollouts = policy_rollout(num_traj=N, env=env, policy=self.policy,
                                      fitted_model=model, eval_mode=False, horizon=horizon,
                                      init_state=init_states, seed=None)
            self.env.env.env.compute_path_rewards(rollouts)
            num_traj, horizon, state_dim = rollouts['observations'].shape
            for i in range(num_traj):
                path = dict()
                obs = rollouts['observations'][i, :, :]
                act = rollouts['actions'][i, :, :]
                rew = rollouts['rewards'][i, :]
                path['observations'] = obs
                path['actions'] = act
                path['rewards'] = rew
                path['terminated'] = False
                paths.append(path)

        # NOTE: If tasks have termination condition, we will assume that the env has
        # a function that can terminate paths appropriately.
        # Otherwise, termination is not considered.

        try:
            paths = self.env.env.env.truncate_paths(paths)
        except AttributeError:
            pass

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    def get_action(self, observation):
        if self.refine is False:
            return self.policy.get_action(observation)
        else:
            return self.get_refined_action(observation)

    def get_refined_action(self, observation):
        # TODO(Aravind): Implemenet this
        # This function should rollout many trajectories according to the learned
        # dynamics model and the policy, and should refine around the policy by
        # incorporating reward based refinement
        raise NotImplementedError
