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
from matplotlib import rcParams


def moving_ave(x):
    res = []
    for i in range(len(x)):
        start_p = max(i-100,0)
        end_p = i+1
        mean = sum(x[start_p:end_p])/len(x[start_p:end_p])
        res.append(mean)
    return res
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

def plot_lc(lc,figure_num=3,save=False):
    plt.figure(figure_num)
    plt.xlabel('Steps')
    plt.ylabel('Lane Change Numbers per Episode')
    X = (np.arange(len(lc))+1)*100
    plt.plot(X,lc,color='blue')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        plt.savefig('success_rate')
def plot_success_rate(success_rate,figure_num2=2,save=False):
    plt.figure(figure_num2)
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    X = (np.arange(len(success_rate))+1)*100
    plt.plot(X,success_rate,color='blue')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        plt.savefig('success_rate')

def plot_offline_rewards(mean_return_list,max_return_list=[],min_return_list=[],std_return_list=[],save=False,figure_num=1):
    plt.figure(figure_num)
    plt.xlabel('Steps')
    plt.ylabel('Average Mean Rewards')
    X = (np.arange(len(mean_return_list))+1)*100
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
def reading(method,has_speed = True,speed_type='cnn'):
    max_index = 549
    root_path = f"/home/i/sacd/sacd/{method}_current_model/"
    if speed_type=='lstm':
        root_path = f"/home/i/sacd/sacd/{method}_lstm_current_model/"
    elif has_speed==False:
        root_path = f"/home/i/sacd/sacd/{method}_no_spd_current_model/"
    # Success Rate
    suc_list = []
    k = 0
    with open(root_path+"eval_suc_rate.txt","r") as file:
        for line in file:
            k = k+1
            if k>max_index:
                break
            suc = [int(x) for x in line.split()]
            suc_list.append(suc)
    suc_rate = []
    for row in suc_list:
        row_mean = sum(row) / len(row)
        suc_rate.append(row_mean)
    transposed_data_sc = [[row[i] for row in suc_list] for i in range(len(suc_list[0]))]
    # Lane Change Numbers
    lc_list = []
    k=0
    with open(root_path+"eval_lc.txt","r") as file:
        for line in file:
            k = k+1
            if k>max_index:
                break
            lc = [int(x) for x in line.split()]
            lc_list.append(lc)
    lc_num = []
    for row in lc_list:
        k = k+1
        if k>max_index:
            break
        row_mean = sum(row) / len(row)
        lc_num.append(row_mean)
    transposed_data_lc = [[row[i] for row in lc_list] for i in range(len(lc_list[0]))]    
    data = []
    # Open the text file for reading
    k=0
    with open(root_path+'eval_data.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            k = k+1
            if k>max_index:
                break
            # Split the line into individual numbers and convert them to integers
            numbers = [float(x) for x in line.split()]
            
            # Append the list of numbers to the 2D list
            # if method=='dqn':
            #     numbers = [i * 1.5 for i in numbers]
            data.append(numbers)
    #data = data[0:300]
    means = []
    k = -1
    skip_id = [0,1]
    for row in data:
        k = k+1
        if k==skip_id:
            continue;
        row_mean = sum(row) / len(row)
        means.append(row_mean)
    #plot_offline_rewards(means)
    transposed_data = [[row[i] for row in data] for i in range(len(data[0]))]
    
    means = means[0:-1]
    suc_rate = suc_rate[0:-1]
    means = moving_ave(means)
    suc_rate = moving_ave(suc_rate)
    x_data = np.arange(len(means))*100
    y_data = smooth(transposed_data, 5)
    y_data_sc = smooth(transposed_data_sc, 5)
    y_data_lc = smooth(transposed_data_lc, 5)
    return x_data,means,suc_rate,y_data,y_data_sc,y_data_lc

def main():
    x_data_d,means_d,suc_rate_d,y_data_d,y_data_d_sc,y_data_d_lc = reading('dqn',speed_type='lstm')
    x_data,means,suc_rate,y_data,y_data_sc,y_data_lc = reading('sacd',speed_type='lstm')
    #x_data_ac,means_ac,suc_rate_ac,y_data_ac,y_data_sc_ac = reading('ac')
    x_data_ns,means_ns,suc_rate_ns,y_data_ns,y_data_ns_sc,y_data_ns_lc = reading('sacd',has_speed=False)
    # plt.figure(1)
    # plt.plot(x_data,means,color='red')
    # plt.xlabel('Steps')
    # plt.ylabel('Average Mean Return')
    # plt.show()
    # plt.figure(2)
    # plt.plot(x_data,suc_rate,color='blue')
    # plt.xlabel('Steps')
    # plt.ylabel('Success Rate')
    #plt.show()
    
    x_data = (np.arange(len(y_data[0]))+1)*100
    x_data_d = (np.arange(len(y_data_d[0]))+1)*100
    #x_data_ac = (np.arange(len(y_data_ac[0]))+1)*1000
    x_data_ns = (np.arange(len(y_data_ns[0]))+1)*100
    sns.set(style="whitegrid", font_scale=1.0,font='Times New Roman')

    plt.figure(figsize=(19.4/2.54, 12/2.54))
    color = ['r', 'g', 'b', 'k']
    linestyle = ['-', '--', ':', '-.']
    sns.tsplot(time=x_data, data=y_data, color=color[0], linestyle=linestyle[0])
    sns.tsplot(time=x_data_d, data=y_data_d, color=color[1], linestyle=linestyle[0])
    #sns.tsplot(time=x_data_ac, data=y_data_ac, color=color[2], linestyle=linestyle[0])
    sns.tsplot(time=x_data_ns, data=y_data_ns, color=color[2], linestyle=linestyle[0])
    plt.xlim(0,55000)
    plt.ylim(30,60) 
    plt.xlabel('Steps')
    plt.ylabel('Average Mean Return')
    plt.legend(['SAC','DQN','SAC-no-Speed'], loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')  # Set bottom spine color to black
    ax.spines['top'].set_color('black')     # Set top spine color to black
    ax.spines['left'].set_color('black')    # Set left spine color to black
    ax.spines['right'].set_color('black')
    plt.savefig("Rewards.png")
    plt.savefig("Rewards.pdf",format="pdf")
    
    plt.figure(figsize=(19.4/2.54, 12/2.54))
    sns.tsplot(time=x_data, data=y_data_sc, color=color[0], linestyle=linestyle[0])
    sns.tsplot(time=x_data_d, data=y_data_d_sc, color=color[1], linestyle=linestyle[0])
    #sns.tsplot(time=x_data_ac, data=y_data_sc_ac, color=color[2], linestyle=linestyle[0])
    sns.tsplot(time=x_data_ns, data=y_data_ns_sc, color=color[2], linestyle=linestyle[0])
    plt.xlim(0,55000)
    plt.ylim(0.4,1.2)
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.legend(['SAC','DQN','SAC-no-Speed'], loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')  # Set bottom spine color to black
    ax.spines['top'].set_color('black')     # Set top spine color to black
    ax.spines['left'].set_color('black')    # Set left spine color to black
    ax.spines['right'].set_color('black')
    plt.savefig("Success_rate.png")
    plt.savefig("Success_rate.pdf",format="pdf")

    plt.figure(figsize=(19.4/2.54, 12/2.54))
    sns.tsplot(time=x_data, data=y_data_lc, color=color[0], linestyle=linestyle[0])
    sns.tsplot(time=x_data_d, data=y_data_d_lc, color=color[1], linestyle=linestyle[0])
    #sns.tsplot(time=x_data_ac, data=y_data_sc_ac, color=color[2], linestyle=linestyle[0])
    sns.tsplot(time=x_data_ns, data=y_data_ns_lc, color=color[2], linestyle=linestyle[0])
    plt.xlim(0,55000)
    plt.ylim(0,8)
    plt.xlabel('Steps')
    plt.ylabel('Average Lane Change Numbers')
    plt.legend(['SAC','DQN','SAC-no-Speed'], loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')  # Set bottom spine color to black
    ax.spines['top'].set_color('black')     # Set top spine color to black
    ax.spines['left'].set_color('black')    # Set left spine color to black
    ax.spines['right'].set_color('black')
    plt.savefig("Lane_change.png")
    plt.savefig("Lane_change.pdf",format="pdf")

    plt.show()

if __name__=="__main__":
    main()