import os
import yaml
import argparse
from datetime import datetime

import gymnasium as gym
from sacd.agent import SacdAgent


def evaluate(args):
    # with open(args.config) as f:
    #     config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = gym.make("myenv",  render_mode='rgb_array')

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent
    agent = Agent(
        env=env, test_env=env,log_dir=log_dir, cuda=False,
        seed=args.seed)
    agent.RENDER = False
    #agent.evaluate(MPC_certi=True)
    agent.offline_eval()
    #agent.MPC_eval()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='myenv')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2**31-1)
    args = parser.parse_args()
    evaluate(args)
