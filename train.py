'''
Author: asdleng lengjianghao2006@163.com
Date: 2023-09-29 10:42:41
LastEditors: asdleng lengjianghao2006@163.com
LastEditTime: 2023-09-29 12:19:46
FilePath: /highway/dsac/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import yaml
import argparse
from datetime import datetime

import gymnasium as gym
from sacd.agent import SacdAgent


def run(args):
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
        env=env, test_env=env, log_dir=log_dir, cuda=True,
        seed=args.seed)
    agent.RENDER = False
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='myenv')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
