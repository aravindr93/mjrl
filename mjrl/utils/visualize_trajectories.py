import mujoco_py
import pickle
import time
import gym
import mj_envs
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='visualize a dataset of trajectories')
    parser.add_argument('env', type=str, help='environemnt id')
    parser.add_argument('paths', type=str, help='saved trajectories')
    parser.add_argument('-L', '--loops', type=int, help='num loops', default=-1)
    args = parser.parse_args()

    env = gym.make(args.env)

    trajectories = pickle.load(open(args.paths, 'rb'))
    # trajectories = [trajectories[0]]

    loops = 0

    while loops < args.loops or args.loops == -1:
        
        for i, trajectory in enumerate(trajectories):
            env.reset()
            init_state = env.env.get_env_state()
            print('trajectory', i)
            for t in range(len(trajectory['qpos'])):
                env.env.mujoco_render_frames = True

                for key in init_state.keys():
                    init_state[key] = trajectory[key][t]

                env.set_env_state(init_state)
                env.step(trajectory['actions'][t])
                env.render()
                # print(trajectory['returns'][t])

            env.env.mujoco_render_frames = False

