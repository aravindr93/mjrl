
import mjrl.envs
import argparse
import pickle

from mjrl.utils.gym_env import GymEnv
from mjrl.utils.replay_buffer import TrajectoryReplayBuffer
from mjrl.samplers.core import sample_paths


if __name__ == '__main__':
    RANDOM_MODE = 'random'
    POLICY_MODE = 'policy'
    MODES = [RANDOM_MODE, POLICY_MODE]

    parser = argparse.ArgumentParser(description='generate a dataset of trajectories')

    parser.add_argument('env', type=str, help='environment id')
    parser.add_argument('-K', '--num-trajs', type=int, default=1)
    parser.add_argument('-M', '--mode', type=str, default=RANDOM_MODE, choices=MODES)
    parser.add_argument('-P', '--policy', type=str)
    parser.add_argument('-o', '--output-dir', type=str, default='./paths.pickle')
    parser.add_argument('-E', '--eval-mode', type=bool, default=False, choices=[True, False]) # TODO: this is not right.
    
    args = parser.parse_args()

    print(args)

    e = GymEnv(args.env)

    if args.mode == POLICY_MODE and args.policy is None:
        print('must specify a policy directory if mode is', POLICY_MODE)
        exit()
    
    replay_buffer = TrajectoryReplayBuffer()

    if args.mode == POLICY_MODE:
        policy = pickle.load(open(args.policy, 'rb'))
    elif args.mode == RANDOM_MODE:
        class RandPolicy():
            def __init__(self, e):
                self.e = e
            def get_action(self, o):
                a = self.e.action_space.sample()
                return [a, {'mean': a, 'evaluation': a}]

        policy = RandPolicy(e)
    paths = sample_paths(args.num_trajs, e, policy, eval_mode=args.eval_mode)
    
    pickle.dump(paths, open(args.output_dir, 'wb'))




