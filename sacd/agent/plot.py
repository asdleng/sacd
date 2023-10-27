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
import numpy as np
import seaborn as sns

# def plot_offline_rewards(collision_rate_list,num,save=False):
#     plt.figure(1)
#     plt.xlabel('Episode')
#     plt.ylabel('Rewards')
#     X = (np.arange(len(collision_rate_list))+1)*10
#     #plt.plot(X,min_return_list,color='blue')
#     plt.plot(X,collision_rate_list,color='blue')
#     #plt.plot(X,max_return_list,color='blue')
#     plt.fill_between(X, min_return_list, max_return_list, color='lavender')

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if save:
#         plt.savefig('eval_rewards')
def plot_success_rate(success_rate,figure_num2=2,save=False):
    plt.figure(figure_num2)
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    X = (np.arange(len(success_rate)))*100
    plt.plot(X,success_rate,color='blue')
    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        plt.savefig('success_rate')

def plot_offline_rewards(mean_return_list,max_return_list=[],min_return_list=[],std_return_list=[],save=False,figure_num=1):
    plt.figure(figure_num)
    plt.xlabel('Steps')
    plt.ylabel('Average Mean Rewards')
    X = (np.arange(len(mean_return_list)))*100
    #plt.plot(X,min_return_list,color='blue')
    plt.plot(X,mean_return_list,color='blue')
    #plt.errorbar(X, mean_return_list, yerr=std_return_list, fmt='o-', capsize=5)

    #plt.plot(X,max_return_list,color='blue')
    #plt.fill_between(X, min_return_list, max_return_list, color='lavender')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        plt.savefig('eval_rewards')

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
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1).to(dtype)
        means = torch.cat((rewards_t[0:99].to(dtype), means))
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
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1).to(dtype)
        means = torch.cat((durations_t[0:99].to(dtype), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are update

def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

def main():
    # Initialize an empty 2D list
    data = []

    # Open the text file for reading
    with open('/home/i/sacd/sacd/sacd_current_model/eval_data.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into individual numbers and convert them to integers
            numbers = [float(x) for x in line.split()]
            
            # Append the list of numbers to the 2D list
            data.append(numbers)

    means = []
    for row in data:
        row_mean = sum(row) / len(row)
        means.append(row_mean)
    #plot_offline_rewards(means)
    transposed_data = [[row[i] for row in data] for i in range(len(data[0]))]

    y_data = smooth(transposed_data, 20)
    x_data = (np.arange(len(y_data[0]))+1)*100
    sns.set(style="darkgrid", font_scale=1.5)
    color = ['r', 'g', 'b', 'k']
    label = ['algo1', 'algo2', 'algo3', 'algo4']
    linestyle = ['-', '--', ':', '-.']
    sns.tsplot(time=x_data, data=y_data, color=color[2], linestyle=linestyle[0])
    plt.savefig("smoothed_figure")
    plt.show()
    print("fuck")
if __name__=="__main__":
    main()