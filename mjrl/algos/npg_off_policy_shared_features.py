
from mjrl.algos.npg_cg import NPG
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.networks import QNetwork
from mjrl.utils.cg_solve import cg_solve
from mjrl.utils.evaluate_q_function import evaluate_n_step, evaluate_start_end, mse

import mjrl.samplers.core as trajectory_sampler
import mjrl.utils.process_samples as process_samples
import numpy as np
import time
import torch

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class NPGOffPolicySharedFeatures(NPG):

    def __init__(self, env, policy, baseline, normalized_step_size,
                    num_policy_updates,
                    num_update_states,
                    num_update_actions,
                    fit_on_policy,
                    fit_off_policy,
                    summary_writer=None,
                    fit_initial=True,
                    ):
        """
        params:
            env
            policy
            baseline
            normalized_step_size
            num_policy_updates
            num_update_states
            num_update_actions
            fit_on_policy
            fit_off_policy
            simple_value_func - whether to estimate V(s) simply or correctly:
                If true, we estimate V(s) = Q(s,a), a = argmax a' pi(a'|s)
                If false, we estimate V(s) = 1/num_value_states sum_i Q(s,a_i), a_i ~ pi(.|s)
            num_value_states
        """
        super().__init__(env, policy, baseline, normalized_step_size = normalized_step_size)
        self.num_policy_updates = num_policy_updates
        self.num_update_states = num_update_states
        self.num_update_actions = num_update_actions
        self.fit_on_policy = fit_on_policy
        self.fit_off_policy = fit_off_policy
        self.summary_writer = summary_writer

        if fit_initial and len(self.baseline.replay_buffers) > 0:
            self.initial_losses = self.baseline.update_returns() # what do we do with this info? log it?


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

        def predict_and_plot(path, i):
            print('plt backend', plt.get_backend())
            obs = torch.from_numpy(path['observations']).float().to(self.policy.device)
            time = torch.from_numpy(path['time']).float().to(self.policy.device)
            values_hat = self.baseline.value(obs, time)

            vh = values_hat.cpu().detach().squeeze().numpy()
            returns = path['returns']

            line_vh, = plt.plot(vh, label='estimated values')
            line_ret, = plt.plot(returns, label='discounted returns')
            plt.legend([line_vh, line_ret])
            plt.savefig('/tmp/returns_and_preds_{}.png'.format(i))
            plt.clf()

        # push samples to rb
        new_rb = TrajectoryReplayBuffer(device=self.policy.device)
        new_rb.push_many(paths)

        self.baseline.new_replay_buffer(new_rb, copy_latest=True)  #TODO: Try with and without this
        return_loss_stats = self.baseline.update_returns()
        # bstats = self.baseline.update_bellman_both_regularized()

        self.baseline.policy_linear.load_state_dict(self.baseline.linear_q_weights[-1].state_dict())
        self.baseline.target_policy_linear.load_state_dict(self.baseline.linear_q_weights[-1].state_dict())

        [predict_and_plot(paths[i], i) for i in range(min(len(paths), 5))]

        # loop over number of policy updates
        update_stats = []
        print('K, rbs', self.baseline.K, len(self.baseline.replay_buffers))
        baseline_stats = []

        total_losses = []
        bellman_losses = []
        reg_losses = []

        update_critic_time = 0.0
        update_policy_time = 0.0
        for k in range(self.num_policy_updates):
            # fit the Q function
            start = time.time()
            #TODO: if k=0 update all?
            bstats = self.baseline.update_bellman_both_regularized() # TODO: uncomment this
            # linear_losses = self.baseline.update_bellman_only_linear()

            # print('linear_losses end', linear_losses[-1][10:])

            update_critic_time += time.time() - start
            # update the policy
            start = time.time()
            stat = self.update_policy(paths, self.fit_on_policy and k == 0)
            update_policy_time += time.time() - start
            
            update_stats.append(stat)

            baseline_stats.append(bstats)
            total_losses.append(bstats['losses'])
            bellman_losses.append(bstats['bellman_losses'])
            reg_losses.append(bstats['reg_losses'])

            # total_losses.append(linear_losses)
        
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)

        if self.summary_writer:
            self.summary_writer.add_scalar('MeanReturn/train', mean_return, iteration)
            self.summary_writer.add_scalar('MinReturn/train', min_return, iteration)
            self.summary_writer.add_scalar('MaxReturn/train', max_return, iteration)
            self.summary_writer.add_scalar('StdReturn/train', std_return, iteration)

            self.summary_writer.add_scalar('Baseline/NumBuffers', len(self.baseline.replay_buffers), iteration)
            self.summary_writer.add_scalar('Baseline/MeanReturnLoss', np.mean(return_loss_stats['losses']), iteration)
            self.summary_writer.add_scalar('Baseline/MeanReturnLoss', np.mean(bstats['losses']), iteration)
            self.summary_writer.add_scalar('Baseline/MeanBellmanLoss', np.mean(bstats['bellman_losses']), iteration)
            self.summary_writer.add_scalar('Baseline/MeanRegLoss', np.mean(bstats['reg_losses']), iteration)

            # self.summary_writer.add_scalar('BufferSize', self.baseline.buffer['observations'].shape[0], iteration)
            self.summary_writer.add_scalar('Time/BellmanUpdate', update_critic_time, iteration)
            self.summary_writer.add_scalar('Time/PolicyUpdate', update_policy_time, iteration)


            for i, stat in enumerate(update_stats):
                alpha, n_step_size, t_gLL, t_FIM, surr_before, surr_after, kl_dist = stat
                self.summary_writer.add_scalar('alpha/sub_iteration_{}'.format(i), alpha, iteration)
                self.summary_writer.add_scalar('delta/sub_iteration_{}'.format(i), n_step_size, iteration)
                self.summary_writer.add_scalar('time_vpg/sub_iteration_{}'.format(i), t_gLL, iteration)
                self.summary_writer.add_scalar('time_npg/sub_iteration_{}'.format(i), t_FIM, iteration)
                self.summary_writer.add_scalar('surr_improvement/sub_iteration_{}'.format(i), surr_after - surr_before, iteration)
                self.summary_writer.add_scalar('kl_dist/sub_iteration_{}'.format(i), kl_dist, iteration)

            total_means = []
            bellman_means = []
            reg_means = []
            for i, (total_loss, bellman_loss, reg_loss) in enumerate(
                zip(total_losses, bellman_losses, reg_losses)):

                mean_total_loss = self._log_stats('TotalLoss', i, iteration, total_loss)
                total_means.append(mean_total_loss)

                mean_bellman_loss = self._log_stats('BellmanLoss', i, iteration, bellman_loss)
                bellman_means.append(mean_bellman_loss)

                mean_reg_loss = self._log_stats('ReturnsLoss', i, iteration, reg_loss)
                reg_means.append(mean_reg_loss)
            # for i, total_loss in enumerate(total_losses):

            #     mean_total_loss = self._log_stats('TotalLoss', i, iteration, total_loss)
            #     total_means.append(mean_total_loss)


            self.summary_writer.add_scalar('MeanTotalLoss/mean', np.mean(total_means), iteration)
            self.summary_writer.add_scalar('MeanBellmanLoss/mean', np.mean(bellman_means), iteration)
            self.summary_writer.add_scalar('MeanReturnsLoss/mean', np.mean(reg_means), iteration)
            
            # log the policy stds
            stds = np.exp(self.policy.log_std.cpu().detach().numpy())
            for i, std in enumerate(stds):
                self.summary_writer.add_scalar('PolicyStd/std_{}'.format(i), std, iteration)

            # mse between predicted q and mc rollouts
            pred_1, mc_1 = evaluate_n_step(1, gamma, paths, self.baseline)
            pred_end, mc_end = evaluate_start_end(gamma, paths, self.baseline)

            self.summary_writer.add_scalar('QFunctionMCMSE_single', mse(pred_1, mc_1), iteration)
            self.summary_writer.add_scalar('QFunctionMCMSE_end', mse(pred_end, mc_end), iteration)

        return [mean_return, std_return, min_return, max_return]

    def _log_stats(self, key, i, iteration, losses):
        mean_loss = np.mean(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        
        self.summary_writer.add_scalar('Mean{}/sub_iteration_{}'.format(key, i), mean_loss, iteration)
        self.summary_writer.add_scalar('Max{}/sub_iteration_{}'.format(key, i), max_loss, iteration)
        self.summary_writer.add_scalar('Min{}/sub_iteration_{}'.format(key, i), min_loss, iteration)
        return mean_loss

    def update_policy(self, paths, fit_on_policy):
        print('update_policy, fit_on_policy:', fit_on_policy)
        if fit_on_policy:
            observations, actions, weights = self.process_paths(paths)
        else:
            observations, actions, weights = self.process_replay()

        # TODO: more normalize weight modes
        print('pre weights mean, std, min, max', np.mean(weights), np.std(weights), np.min(weights), np.max(weights))
        weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-6) # TODO: uncomment this?
        print('post weights mean, std, min, max', np.mean(weights), np.std(weights), np.min(weights), np.max(weights))

        # import ipdb; ipdb.set_trace()

        # update policy
        surr_before = self.CPI_surrogate(observations, actions, weights).data.cpu().numpy().ravel()[0]
        alpha, n_step_size, t_gLL, t_FIM, new_params, old_params, vpg_grad = self.update_from_states_actions(observations, actions, weights)
        
        if np.isnan(new_params).any():
            print(np.isnan(vpg_grad).any(), (vpg_grad == 0).all())
            import ipdb; ipdb.set_trace()
            raise RuntimeError('policy has nan params')

        surr_after = self.CPI_surrogate(observations, actions, weights).data.cpu().numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.cpu().numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)
        return alpha, n_step_size, t_gLL, t_FIM, surr_before, surr_after, kl_dist

    def update_from_states_actions(self, observations, actions, weights):
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------

        # VPG
        ts = time.time()
        vpg_grad = self.flat_vpg(observations, actions, weights)
        vpg_cpy = vpg_grad.copy()
        t_gLL += time.time() - ts

        # NPG
        ts = time.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += time.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)

        return alpha, n_step_size, t_gLL, t_FIM, new_params, curr_params, vpg_cpy

    def process_paths(self, paths):

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        times = np.concatenate([path["time"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        obs_t = torch.from_numpy(observations).to(self.baseline.device).float()
        # act_t = torch.from_numpy(actions).to(self.baseline.device).float()
        time_t = torch.from_numpy(times).to(self.baseline.device).float()
        # weights = self.baseline.advantage(obs_t, act_t, time_t).cpu().detach().numpy()
        # weights = self.baseline.advantage(obs_t, act_t, time_t, use_mu_approx=False, n_value=10).cpu().detach().numpy()
        # weights = self.baseline.q_value(obs_t, act_t, time_t).cpu().detach().numpy()

        # weights = returns - self.baseline.value(obs_t, time_t, use_mu_approx=False, n_value=4).cpu().detach().numpy()
        weights = returns - self.baseline.value(obs_t, time_t).cpu().detach().numpy()


        return observations, actions, weights
    
    def process_replay(self):
        # sample states from newest data only TODO: turn this off
        # p = [0.0] * self.baseline.K
        # p[-1] = 1.0
        # samples, _ = self.baseline.sample_replay_buffers_efficient(self.num_update_states, p=p)
        samples, _ = self.baseline.sample_replay_buffers_efficient(self.num_update_states)

        observations = samples['observations']
        times = samples['time']
        # actions_all = torch.zeros(self.num_update_states * self.num_update_actions, self.env.action_dim)
        # for i in range(self.num_update_actions):
        #     actions = self.policy.get_action_pytorch(observations)
        #     actions_all[i * self.num_update_states:(i + 1) * self.num_update_states] = actions

        actions_all_list = []
        for i in range(self.num_update_actions):
            actions = self.policy.get_action_pytorch(observations)
            actions_all_list.append(actions)

        actions_all = torch.cat(actions_all_list)
        obs_tiled = observations.repeat(self.num_update_actions, 1)
        times_tiled = times.repeat(self.num_update_actions)
        
        # weights = self.baseline.advantage(obs_tiled, actions_all, times_tiled) # TODO: pass use_mu_approx
        Q_values = self.baseline.q_value(obs_tiled, actions_all, times_tiled)
        values = torch.zeros((observations.shape[0], 1)).to(self.policy.device)
        for i in range(self.num_update_actions):
            values += Q_values[i * self.num_update_states:(i + 1) * self.num_update_states]
            
        values_tiled = values.repeat(self.num_update_actions, 1) / self.num_update_actions

        weights = Q_values - values_tiled

        return obs_tiled.detach().cpu().numpy(), actions_all.detach().cpu().numpy(), weights.detach().cpu().numpy()