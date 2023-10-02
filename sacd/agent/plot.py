'''
Author: asdleng lengjianghao2006@163.com
Date: 2023-09-27 14:04:25
LastEditors: asdleng lengjianghao2006@163.com
LastEditTime: 2023-09-27 14:30:52
FilePath: /highway/hybridsac/plot.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import torch

def plot_rewards(episode_rewards,show_result=False):
    plt.figure(1)
    dtype=torch.float
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    #cost_t = torch.tensor(cost_list, dtype=torch.float)
    if show_result:
        plt.title('Rewards')
        plt.savefig('Rewards')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy(),color='red')
    if len(rewards_t) >= 20:
        means = rewards_t.unfold(0, 20, 1).mean(1).view(-1).to(dtype)
        means = torch.cat((rewards_t[0:19].to(dtype), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_durations(episode_durations,show_result=False):
    plt.figure(3)
    dtype=torch.float
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    #cost_t = torch.tensor(cost_list, dtype=torch.float)
    if show_result:
        plt.title('Durations')
        plt.savefig('Durations')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    plt.plot(durations_t.numpy(),color='red')
    if len(durations_t) >= 20:
        means = durations_t.unfold(0, 20, 1).mean(1).view(-1).to(dtype)
        means = torch.cat((durations_t[0:19].to(dtype), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated