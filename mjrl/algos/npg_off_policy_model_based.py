
from mjrl.algos.npg_cg import NPG
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QNetwork
from mjrl.utils.cg_solve import cg_solve
from mjrl.utils.evaluate_q_function import evaluate_n_step, evaluate_start_end, mse

from mjrl.utils.process_samples import discount_sum

import mjrl.samplers.core as trajectory_sampler
import mjrl.utils.process_samples as process_samples
import numpy as np
import time
import torch

from mjrl.algos.model_accel.sampling import policy_rollout

NO_VALUE = 'NO_VALUE'

class NPGOffPolicyModelBased(NPG):

    def __init__(self, env, policy, baseline, normalized_step_size, H, num_update_paths,
        num_policy_updates, gae_lambda=None, normalize_advantage=False, summary_writer=None,
        mode=NO_VALUE):
        super().__init__(env, policy, baseline, normalized_step_size=normalized_step_size)
        # assume baseline has replay_buffer, fitted_model, gamma, 

        self.T = env.horizon
        self.H = H
        self.num_update_paths = num_update_paths
        self.num_policy_updates = num_policy_updates
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.summary_writer = summary_writer

        self.mode = mode
        print('MODE:', self.mode)

    def train_step(self, N,
                    env=None,
                    sample_mode='trajectories',
                    horizon=1e6,
                    gamma=0.995,
                    gae_lambda=0.97,
                    num_cpu='max',
                    env_kwargs=None,
                    iteration=None):

        if iteration is None:
            raise Exception('Must set include_iteration=True in train_step()')

        # get new samples
        env = self.env.env_id if env is None else env
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                                base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_paths(**input_dict)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                                base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_data_batch(**input_dict)

        # compute returns
        process_samples.compute_returns(paths, gamma)

        # push samples to rb
        self.baseline.replay_buffer.push_many(paths)

        # since we have the on policy data, let us use it
        # returns_loss = self.baseline.fit_returns(paths, epochs=20)
        # print('returns loss', returns_loss)

        stats = self.baseline.fit_on_policy(paths, 64, 500) # TODO: tune

        print('on_policy')
        print('average total loss', stats['sum_total'])
        print('average recon loss', stats['sum_recon'])
        print('average reward loss', stats['sum_reward'])
        print('average value loss', stats['sum_value'])

        print('last total loss', stats['last_total'])
        print('last recon loss', stats['last_recon'])
        print('last reward loss', stats['last_reward'])
        print('last value loss', stats['last_value'])
        print()

        # if False:
        #     all_epoch_losses = self.baseline.update_all_models(mb_size=64, epochs=10)

        #     print('updade all models last losses', [losses[-1] for losses in all_epoch_losses])
        #     print('updade all models avg losses', [np.mean(losses) for losses in all_epoch_losses])
        # else:
        #     all_stats = self.baseline.update()
        #     print('average total loss', np.mean([stats['sum_total'] for stats in all_stats]))
        #     print('average recon loss', np.mean([stats['sum_recon'] for stats in all_stats]))
        #     print('average reward loss', np.mean([stats['sum_reward'] for stats in all_stats]))
        #     print('average value loss', np.mean([stats['sum_value'] for stats in all_stats]))

        rollout_time = 0.0
        update_baseline_time = 0.0
        fit_returns_time = 0.0
        update_policy_time = 0.0
        
        # loop over number of policy updates
        for k in range(self.num_policy_updates):
            
            start = time.time()
            ficticious_paths = self.rollout_ficticious()
            process_samples.compute_returns(ficticious_paths, gamma)
            # self.baseline.replay_buffer.push_many_temp(ficticious_paths)
            rollout_time += time.time() - start

            start = time.time()
            # update baseline
            all_stats = self.baseline.update()
            update_baseline_time += time.time() - start


            # update policy
            # self.update_policy()
            start = time.time()
            self.update_from_mb_rollout(ficticious_paths)
            update_policy_time += time.time() - start
            # self.baseline.replay_buffer.pop_temp()

            if self.mode == NO_VALUE:
                start = time.time()
                ep_losses = self.baseline.fit_returns(ficticious_paths, epochs=5)
                print('baseline value loss', ep_losses)
                fit_returns_time += time.time() - start

            # all_stats = self.baseline.update_traj()
            print('average total loss', np.mean([stats['sum_total'] for stats in all_stats]))
            print('average recon loss', np.mean([stats['sum_recon'] for stats in all_stats]))
            print('average reward loss', np.mean([stats['sum_reward'] for stats in all_stats]))
            print('average value loss', np.mean([stats['sum_value'] for stats in all_stats]))

        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)

        # print('rollout_time', rollout_time)
        # print('update_baseline_time', update_baseline_time)
        # print('fit_returns_time', fit_returns_time)
        # print('update_policy_time', update_policy_time)
        
        # rollout_time 3.230736494064331
        # update_baseline_time 148.9973521232605
        # fit_returns_time 42.80794668197632
        # update_policy_time 7.572345733642578


        return [mean_return, std_return, min_return, max_return, N]
    
    # def update_policy(self):
    #     # two options. TODO: try both
    #     # self.update_from_rb()
    #     self.update_from_mb_rollout(paths)

    def update_from_rb(self):
        pass

    def rollout_ficticious(self):
        # get mix of init states from start state dist and all states
        # sample all states
        # samples = self.baseline.replay_buffer.get_sample(self.num_update_paths // 2)
        samples = self.baseline.replay_buffer.get_sample_starting_at_or_before(self.num_update_paths // 2,
            t=self.T - self.H)

        init_sampled = self.baseline.replay_buffer.sample_initial_states(self.num_update_paths // 2)
        non_init_sampled = [x.cpu().numpy() for x in samples['observations']] 
        init_states = non_init_sampled + init_sampled

        init_times = [x.cpu().numpy() for x in samples['time']] + list(np.zeros(len(init_sampled)))

        paths = []
        for model in self.baseline.fitted_model:
            # dont set seed explicitly -- this will make rollouts follow tne global seed
            rollouts = policy_rollout(num_traj=self.num_update_paths, env=self.env, policy=self.policy,
                                      fitted_model=model, eval_mode=False, horizon=self.H,
                                      init_state=init_states, init_times=init_times, seed=None)
            self.env.env.env.compute_path_rewards(rollouts)
            num_traj, horizon, state_dim = rollouts['observations'].shape
            for i in range(num_traj):
                path = dict()
                obs = rollouts['observations'][i, :, :]
                act = rollouts['actions'][i, :, :]
                rew = rollouts['rewards'][i, :]
                time = rollouts['time'][i,:]
                is_terminal = rollouts['is_terminal'][i,:]
                
                path['observations'] = obs
                path['actions'] = act
                path['rewards'] = rew
                path['terminated'] = False
                path['time'] = time
                path['is_terminal'] = is_terminal
                paths.append(path)

        # NOTE: If tasks have termination condition, we will assume that the env has
        # a function that can terminate paths appropriately.
        # Otherwise, termination is not considered.

        try:
            paths = self.env.env.env.truncate_paths(paths)
        except AttributeError:
            pass
            
        return paths

    def update_from_mb_rollout(self, paths):
        
        # compute returns
        process_samples.compute_returns(paths, self.baseline.gamma)
        # compute advantages
        # process_samples.compute_advantages(paths, self.baseline, self.baseline.gamma, self.gae_lambda)
        self.compute_advantages(paths)
        # train from paths
        eval_statistics = self.train_from_paths(paths)

        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        print('mean ficticiuos return', mean_return)

    def compute_advantages(self, paths):
        print('compute_advantages', self.gae_lambda)
        # assume paths are of length H
        if self.gae_lambda == None:
            if self.mode == NO_VALUE:
                for path in paths:
                    path["baseline"] = self.baseline.predict(path)
                    path["advantages"] = path["returns"] - path["baseline"]
            else:
                for path in paths:
                    path["baseline"] = self.baseline.predict(path)
                    terminal_value = path["baseline"][-1]
                    Hleft = np.ones(self.H) * self.H - np.arange(self.H) - 1
                    # Qs = path["returns"] - path["returns"][-1] + self.baseline.gamma**Hleft * terminal_value
                    Qs = path["returns"] + self.baseline.gamma**Hleft * terminal_value

                    path["advantages"] = Qs - path["baseline"]
                    path["advantages"][-1] = path["returns"][-1]
                    # TODO: check if this is correct!!!!
                    # print('WARNING untested code')
        # GAE mode
        else:
            for path in paths:
                b = path["baseline"] = self.baseline.predict(path)
                if b.ndim == 1:
                    b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
                else:
                    b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
                td_deltas = path["rewards"] + self.baseline.gamma*b1[1:] - b1[:-1]
                path["advantages"] = discount_sum(td_deltas, self.baseline.gamma*self.gae_lambda)
        if self.normalize_advantage:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)