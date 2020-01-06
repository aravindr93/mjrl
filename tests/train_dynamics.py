import pickle
import mjrl
from mjrl.samplers.core import sample_paths
import mjrl.envs
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.TupleMLP import TupleMLP
import torch
from mjrl.baselines.model_based_baseline import MBBaselineNaive
import torch.nn as nn

from mjrl.algos.model_accel.nn_dynamics import DynamicsModel

pol = pickle.load(open('./point_mass_exp1/iterations/policy_10.pickle', 'rb'))

paths = sample_paths(50, 'mjrl_point_mass-v0', pol)

replay_buffer = TrajectoryReplayBuffer()
replay_buffer.push_many(paths)

e = GymEnv('mjrl_point_mass-v0')
state_dim = e.observation_dim
action_dim = e.action_dim

HIDDEN_SIZE = 64
time_dim = 3

# model_network = TupleMLP(state_dim + action_dim, state_dim, (HIDDEN_SIZE, HIDDEN_SIZE), non_linearity=torch.nn.Tanh)
# model_network = nn.Linear(state_dim + action_dim, state_dim)
# params = list(model_network.parameters())
# params[-1].data *= 1e-2
# params[-2].data *= 1e-2

model = DynamicsModel(state_dim=e.observation_dim, act_dim=e.action_dim) 

baseline = MBBaselineNaive(model, nn.Linear(1, 1), nn.Linear(1, 1), pol, None, replay_buffer, 1, e.horizon, 0.9,
        num_model_fit_iters=1000, model_batch_size=64, model_lr=1e-3)
model_loss = baseline.update_model()

def test_single():
    sample = replay_buffer.get_sample()
    pred = model.network(sample['observations'], sample['actions'])

    # print(delta)
    print(pred)
    print(sample['next_observations'])
    print(sample['actions'])
    return sample

test_single();

pickle.dump(model, open('dynamics.pickle', 'wb'))

def make_sure_close_single():
    sample = replay_buffer.get_sample_safe()
    diff = torch.norm(sample['observations'] - sample['next_observations'])
    print(diff)
    return sample

