import gym
import mjrl.envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import MetaWorldEnv
from mjrl.policies.gaussian_mlp import MLP
from metaworld.benchmarks import ML1


DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name mjrl_swimmer-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--task_name', type=str, help='task to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)

def main(task_name, policy, mode, seed, episodes):
    env = ML1.get_train_tasks(task_name)
    e = MetaWorldEnv(env)
    e.set_seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)
    # render policy
    e.visualize_policy(pi, num_episodes=episodes, horizon=e.horizon, mode=mode)

if __name__ == '__main__':
    main()

