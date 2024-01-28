from asyncio import FastChildWatcher
import os
import sys
import io
import numpy as np
import torch
from torch.optim import Adam
import multiprocessing
import gymnasium as gym
from .base import BaseAgent
from sacd.model import TwinnedQNetwork2, CateoricalPolicy2
from sacd.utils import disable_gradients, remove_files_in_directory
from .mpc_new import MPC, MPC2, MPC3,  MPC_des
from .plot import plot_rewards, plot_durations, plot_offline_rewards,plot_success_rate,plot_lc
import time
import random
#from .speed_generator import generate_speed_with_random_acc,normalize_speed
from .speed_gener import normalize_speed,load_spd_data,random_select_spd
class SacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=200000, batch_size=1024,
                 lr=0.0001, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.90, start_steps=101,
                 update_interval=4, target_update_interval=1000,
                 use_per=False, dueling_net=False, num_eval_steps=1000,
                 max_episode_steps=1000, log_interval=100, log_interval_ep=1,eval_interval=10000,
                 cuda=True, seed=0, has_speed=True, spd_type = 'lstm', method='lstm'):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, log_interval_ep, eval_interval, cuda, seed,has_speed,spd_type,method)
        self.spd_sv = load_spd_data()
        self.log_interval_ep = log_interval_ep
        self.has_speed = has_speed
        self.seed = seed
        self.spd_type = spd_type
        self.method = method
        self.num_episodes = 20000
        self.RENDER = False
        self.state_dim = self.env.observation_space.shape[0] * \
            self.env.observation_space.shape[1]
        self.act_dim = 3
        self.epsilon = 1.0
        self.upper = 20
        self.lower = 0
        self.continue_train = True
        self.continue_eval = True
        # 注意，这里是为了避免频繁换道
        self.policy_frequency = 1
        self.env.configure(
            {
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "duration": 500,
                "screen_width": 1000,
                "screen_height": 200,
                "show_trajectories": False,
                "render_agent": False,
                "offscreen_rendering": True,
                'ego_spacing': 1.5
            })
        if (self.RENDER):
            self.env.config['offscreen_rendering'] = False

        self.try_return_list = []
        self.try_success_rate_list = []
        self.try_lc_list = []
        #
        self.episode_rewards = []
        self.episode_durations = []
        # Define networks.
        try: # 可能由于版本问题报错，但发现执行两次就可以了，所以用try一下
            self.policy = CateoricalPolicy2(
                self.state_dim, self.act_dim,has_speed = self.has_speed,spd_type = self.spd_type).to(device=self.device)
        except RuntimeError:
            self.policy = CateoricalPolicy2(
                self.state_dim, self.act_dim,has_speed = self.has_speed,spd_type = self.spd_type).to(device=self.device)
        self.online_critic = TwinnedQNetwork2(
            self.state_dim, self.act_dim,
            dueling_net=dueling_net,has_speed = self.has_speed,spd_type = self.spd_type).to(device=self.device)
        self.target_critic = TwinnedQNetwork2(
            self.state_dim, self.act_dim,
            dueling_net=dueling_net,has_speed = self.has_speed,spd_type = self.spd_type).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.act_dim,) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
    def make_state(self,env,state):
        state = state.reshape([self.state_dim])
        state[1] = (env.vehicle.position[0]-env.start_position)/env.trip_dis
        state[0] = env.time/env.x_time[-1]
        return state
    def make_state0(self,env,state):
        state = state.reshape([self.state_dim])
        return state
    def explore(self, state,speed_seq):
        # Act with randomness.
        state = torch.tensor(state[None, ...]).to(self.device).float()
        speed_seq = torch.tensor(speed_seq[None, ...]).to(self.device).float()
        with torch.no_grad():
            action, _, _ = self.policy.sample(state,speed_seq)
        return action.item()

    def exploit(self, state, speed_seq):
        # Act without randomness.
        state = torch.tensor(state[None, ...]).to(self.device).float()
        speed_seq = torch.tensor(speed_seq[None, ...]).to(self.device).float()
        with torch.no_grad():
            action = self.policy.act(state,speed_seq)
        return action.item()

    def dqn_explore(self, state, speed_seq):
        state = torch.tensor(state[None, ...]).to(self.device).float()
        speed_seq = torch.tensor(speed_seq[None, ...]).to(self.device).float()
        self.update_eps();
        if random.random() < self.epsilon:
            # Explore: Choose a random action
            action = torch.tensor([[random.randint(0, self.act_dim - 1)]], device=self.device)
        else:
            # Exploit: Choose the action with the highest Q-value
            with torch.no_grad():
                q1,q2 = self.online_critic(state, speed_seq) # 出来的是每个动作的Q值
                action = torch.argmax(q1, dim=1, keepdim=True)
        return action.item()
    def dqn_exploit(self, state,speed_seq):
        state = torch.tensor(state[None, ...]).to(self.device).float()
        speed_seq = torch.tensor(speed_seq[None, ...]).to(self.device).float()
        q1,q2 = self.online_critic(state, speed_seq) # 出来的是每个动作的Q值
        action = torch.argmax(q1, dim=1, keepdim=True)
        return action.item()

    def update_eps(self):
        self.epsilon = max(0.01, 1.0 * (0.995 ** self.episodes))

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states,speed_seq1s, actions, rewards, next_states,speed_seq2s, dones):
        curr_q1, curr_q2 = self.online_critic(states,speed_seq1s)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states,speed_seq1, actions, rewards, next_states,speed_seq2, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states,speed_seq2)
            next_q1, next_q2 = self.target_critic(next_states,speed_seq2)
            if self.method=='sacd':
                next_q = (action_probs * (
                    torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                    )).sum(dim=1, keepdim=True)
            elif self.method=='dqn':
                next_q = torch.max(next_q1,1,keepdim=True)[0]
            elif self.method=='ac':
                next_q = (action_probs * (
                torch.min(next_q1, next_q2))).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states,speed_seq1s, actions, rewards, next_states,speed_seq2s, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states,speed_seq1s)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states,speed_seq1s)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        if self.method=='ac':
            policy_loss = (weights * (- q)).mean()
        else:
            policy_loss = (weights * (- q - self.alpha * entropies)).mean()
        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def load_models(self, load_dir):
        self.memory = self.memory.load_replay_buffer(os.path.join(load_dir, 'replay_buffer.pkl'))
        self.online_critic.load(os.path.join(load_dir, 'online_critic.pth'))
        self.target_critic.load(os.path.join(load_dir, 'target_critic.pth'))        
        self.policy.load(os.path.join(load_dir, 'policy.pth'))
        if self.method=='sacd':
            self.log_alpha = torch.load(os.path.join(load_dir,'log_alpha.pth'))
        with open(load_dir+"current_states.txt", "r") as file:
            self.episodes = int(file.readline().strip())
            self.steps = int(file.readline().strip())

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.memory.save_replay_buffer(os.path.join(save_dir,'replay_buffer.pkl'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        if self.method=='sacd':
            torch.save(self.log_alpha,os.path.join(save_dir, 'log_alpha.pth'))
        with open(save_dir+"current_states.txt", "w") as file:
            file.write(str(self.episodes) + "\n")
            file.write(str(self.steps) + "\n")

    def get_MPC_actions(self, env, action_d, train=True, eval_MPC_protect=True, horizon=10, dt=0.1):
        action_g = np.array([0.0, 0.0])
        ego = env.vehicle

        current_state = np.array(
            [ego.position[0], ego.position[1], ego.speed, ego.heading])
        predict_info, surr_vehicle = env.predict(horizon, dt)
        current_lane = round(ego.position[1]/4)
        current_lane = np.clip(current_lane, 0, 2)
        target_lane = np.clip((current_lane + action_d-1), 0, 2)
        reward_no_solution = 0
        no_solution_done = False
        delta_t = 0
        cert = False
        if train:
            # 训练过程，不管换道约束
            if target_lane == current_lane: # 跟车
                status, u_opt, x_opt, obj_value = MPC(
                    env, current_state, target_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
                if not status:
                    status, u_opt, x_opt, obj_value = MPC2(
                        env, current_state, current_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
            else:   # 换道
                status, u_opt, x_opt, obj_value = MPC3(
                    env, current_state, target_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
            action_g = u_opt[0, :]
            if status == False:
                reward_no_solution = -10
                no_solution_done = True
            action_g = u_opt[0, :]
            return action_g, reward_no_solution, no_solution_done,-delta_t,cert
        else:
            # 验证过程，考虑MPC约束
            if eval_MPC_protect:
                time1 = time.time()
                status, u_opt, x_opt, obj_value = MPC(
                    env, current_state, target_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
                time2 = time.time()
                delta_t = time1-time2
            # 如果没有解，就使用跟车模式
                if status == False:
                    print("No Solution from ", current_lane, "to", target_lane)
                    cert = True
                    status, u_opt, x_opt, obj_value = MPC2(
                        env, current_state, current_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
                    action_g[0] = u_opt[0, 0]
                    action_g[1] = u_opt[0, 1]
                    if status == False:
                        reward_no_solution = -10
                        no_solution_done = True
                        print("Totally no Solution!!!")
                        action_g[0] = -5
                        action_g[1] = 0
                else:
                    action_g = u_opt[0, :]
            else:   # 或不考虑
                if target_lane != current_lane:  # 换道
                    time1 = time.time()
                    status, u_opt, x_opt, obj_value = MPC3(
                        env, current_state, target_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
                    time2 = time.time()
                    delta_t = time1-time2
                else:  # 跟车
                    time1 = time.time()
                    status, u_opt, x_opt, obj_value = MPC(
                        env, current_state, current_lane, ego.LENGTH, predict_info, surr_vehicle, horizon, dt)
                    time2 = time.time()
                    delta_t = time1-time2
                action_g = u_opt[0, :]
            return action_g, reward_no_solution, no_solution_done,-delta_t,cert
    def ger_uniform_num(self,low,high):
        r = np.random.random()
        res = low+r*(high-low)
        return res
    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0
        # init_speed = self.ger_uniform_num(self.lower,self.lower+4)
        # mean = 0
        # std = 0.1
        # length = 300
        speed_array = random_select_spd(self.spd_sv,300)
        self.env.set_ref_speed2(speed_array,10)
        self.env.configure(
            {
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "duration": 500,
                "screen_width": 1000,
                "screen_height": 200,
                "show_trajectories": False,
                "render_agent": True,
                "offscreen_rendering": False,
                "initial_lane_id": np.random.choice([0, 1, 2]),
                "vehicles_density": self.ger_uniform_num(1.0, 1.4),
                "ego_spacing":1.5
            })
        done, truncate = False, False
        state = self.env.reset()
        #state = self.make_state0(self.env,state[0])
        state = self.make_state(self.env,state[0])
        eco_seq = self.env.get_eco_seq()
        last_d = 1
        if (self.RENDER == True):
            self.env.render()
        count = 0
        last_lane = self.env.config['initial_lane_id']
        action_d = 1

        while (not done and not truncate) and episode_steps <= self.max_episode_steps:
            current_lane = np.clip(round(self.env.vehicle.position[1]/4), 0, 2)
            spd_seq = normalize_speed(self.env.get_speed_seq(100),self.upper,self.lower)
            if count % round(self.env.config['policy_frequency']/self.policy_frequency) == 0:
                step_reward  = 0
                # 这里才会真正做决策！！！！
                if self.method == 'sacd' or self.method == 'ac':
                    action_d = self.explore(state,eco_seq)
                    #action_d = self.explore(state,spd_seq)
                elif self.method == 'dqn':
                    action_d = self.dqn_explore(state,eco_seq)
                    #action_d = self.dqn_explore(state,spd_seq)
                if count==0:
                    old_state = state
                    old_eco_seq =  self.env.get_eco_seq()
                    #old_spd_seq =  normalize_speed(self.env.get_speed_seq(100),self.upper,self.lower)
                    old_action = action_d
            else:
                # 否则执行既有决策！！！！
                action_d = last_d
            action_g, reward_no_solution, no_solution_done,delta_t,cert = self.get_MPC_actions(self.env,
                action_d, train=True, horizon = 10, dt=0.4)

            next_state, reward, done, truncate, info = self.env.step(action_g)

            if (self.RENDER == True):
                self.env.render()

            last_d = action_d
            reward += reward_no_solution
            step_reward += reward
            done  = done or no_solution_done    # 无解，也是done

            #next_state = self.make_state0(self.env,np.array(next_state))
            next_state = self.make_state(self.env,np.array(next_state))
            next_eco_seq =  self.env.get_eco_seq()
            #next_spd_seq = normalize_speed(self.env.get_speed_seq(100),self.upper,self.lower)
            # Clip reward to [-1.0, 1.0].
            # clipped_reward = max(min(reward, 1.0), -1.0)
            state = next_state
            # To calculate efficiently, set priority=max_priority here.
            if count>0 or done or truncate :
                if count % round(self.env.config['policy_frequency']/self.policy_frequency) == 0 or done or truncate:
                    self.memory.append(old_state,old_eco_seq, old_action, step_reward,
                                     next_state, next_eco_seq, (done or truncate))
                    #self.memory.append(old_state,old_spd_seq, old_action, step_reward,
                    #                next_state, next_spd_seq, (done or truncate))
                    old_state = next_state
                    old_eco_seq = next_eco_seq
                    #old_spd_seq = next_spd_seq
                    old_action = action_d
                    self.steps += 1
                    episode_steps += 1
                    episode_return += step_reward
                    step_reward = 0

            

            if self.is_update() and len(self.memory)>0:
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            count += 1
            # 使用step数存模型
            if self.steps % self.log_interval == 0 and self.steps>0:    
                filename = f"{self.method}_model/model_{self.method.upper()}_steps_{self.steps}.pth"
                if self.spd_type=='lstm':
                    filename = f"{self.method}_lstm_model/model_{self.method.upper()}_steps_{self.steps}.pth"
                elif self.has_speed==False:
                    filename = f"{self.method}_no_spd_model/model_{self.method.upper()}_steps_{self.steps}.pth"
                if self.method!='dqn':  #SAC使用的是\pi网络
                    torch.save(self.policy.state_dict(), filename)
                else:   # DQN使用的是Q网络
                    torch.save(self.online_critic.state_dict(), filename)
                current_save_path = f"/home/i/sacd/sacd/{self.method}_current_model/"
                if self.spd_type=='lstm':
                    current_save_path = f"/home/i/sacd/sacd/{self.method}_lstm_current_model/"
                elif self.has_speed==False:
                    current_save_path = f"/home/i/sacd/sacd/{self.method}_no_spd_current_model/"
                self.save_models(current_save_path)
                
        # We log running mean of training rewards.
        self.train_return.append(episode_return)
        self.episode_rewards.append(episode_return)
        self.episode_durations.append(episode_steps/self.env.config['policy_frequency'])
        # 使用episode数存模型
        # if self.episodes % self.log_interval_ep == 0 and self.steps>0:    
        #     filename = f"{self.method}_model/model_{self.method.upper()}_episodes_{self.episodes}.pth"
        #     if self.spd_type=='lstm':
        #         filename = f"{self.method}_lstm_model/model_{self.method.upper()}_episodes_{self.episodes}.pth"
        #     if self.has_speed==False:
        #         filename = f"{self.method}_no_spd_model/model_{self.method.upper()}_episodes_{self.episodes}.pth"
        #     if self.method!='dqn':  #SAC使用的是\pi网络
        #         torch.save(self.policy.state_dict(), filename)
        #     else:   # DQN使用的是Q网络
        #         torch.save(self.online_critic.state_dict(), filename)
        #     current_save_path = f"/home/i/sacd/sacd/{self.method}_current_model/"
        #     if self.spd_type=='lstm':
        #         current_save_path = f"/home/i/sacd/sacd/{self.method}_lstm_current_model/"
        #     if self.has_speed==False:
        #         current_save_path = f"/home/i/sacd/sacd/{self.method}_no_spd_current_model/"
        #     self.save_models(current_save_path)

        print(f'Episode: {self.episodes:<4}  '
              f'Steps: {self.steps:<4}  '
              f'Return: {episode_return:<5.2f}')

    def run(self):
        if (self.RENDER):
            self.env.config['offscreen_rendering'] = False
        if self.continue_train:
            load_path = f"/home/i/sacd/sacd/{self.method}_current_model/"
            if self.spd_type=='lstm':
                load_path = f"/home/i/sacd/sacd/{self.method}_lstm_current_model/"
            if self.has_speed==False:
                load_path = f"/home/i/sacd/sacd/{self.method}_no_spd_current_model/"
            self.load_models(load_path)
        else:
            # 清空模型文件
            model_path = f"/home/i/sacd/{self.method}_model/"
            if self.spd_type=='lstm':
                model_path = f"/home/i/sacd/{self.method}_lstm_model/"
            if self.has_speed==False:
                model_path = f"/home/i/sacd/{self.method}_no_spd_model/"
            remove_files_in_directory(model_path)
        while True:
            self.train_episode()
            if self.episodes > self.num_episodes:
                break
            if self.steps > self.num_steps:
                break
        # save the policy
        #torch.save(self.policy.state_dict(), 'sacd.pth')


    def offline_eval(self):
        std_return_list = []
        return_list = []
        success_rate_list = []
        lc_list = []
        min_return_list = []
        max_return_list = []
        j = 1
        root_path = f"/home/i/sacd/sacd/{self.method}_current_model/"
        if self.spd_type=='lstm':
            root_path = f"/home/i/sacd/sacd/{self.method}_lstm_current_model/"
        if self.has_speed==False:
            root_path = f"/home/i/sacd/sacd/{self.method}_no_spd_current_model/"
        if self.continue_eval:
            file_path = root_path+"eval_data.txt"
            with open(file_path, 'r') as file:
                # Iterate through each line in the file
                for line in file:
                    # Split the line into individual numbers and convert them to integers
                    numbers = [float(x) for x in line.split()]
                    return_list.append(sum(numbers)/len(numbers))
                    max_return_list.append(max(numbers))
                    min_return_list.append(min(numbers))
                    j = j+1
                    # Append the list of numbers to the 2D list
            file_path = root_path+"eval_suc_rate.txt"
            with open(file_path, 'r') as file:
                # Iterate through each line in the file
                for line in file:
                    # Split the line into individual numbers and convert them to integers
                    numbers = [float(x) for x in line.split()]
                    success_rate_list.append(sum(numbers)/len(numbers))
                    # Append the list of numbers to the 2D list
            file_path = root_path+"eval_lc.txt"
            with open(file_path, 'r') as file:
                # Iterate through each line in the file
                for line in file:
                    # Split the line into individual numbers and convert them to integers
                    numbers = [float(x) for x in line.split()]
                    lc_list.append(sum(numbers)/len(numbers))
                    # Append the list of numbers to the 2D list
        else:
            # 清空文件
            with open(root_path+"eval_data.txt", 'w'):
                pass
            with open(root_path+"eval_suc_rate.txt", 'w'):
                pass
            with open(root_path+"eval_lc.txt", 'w'):
                pass
        while j <= 2000:
            num = str(j)
            file_path = f"/home/i/sacd/{self.method}_model/model_{self.method.upper()}_steps_"+num+"00.pth"
            if self.spd_type=='lstm':
                file_path = f"/home/i/sacd/{self.method}_lstm_model/model_{self.method.upper()}_steps_"+num+"00.pth"
            if self.has_speed==False:
                file_path = f"/home/i/sacd/{self.method}_no_spd_model/model_{self.method.upper()}_steps_"+num+"00.pth"
            try:
                if j > 0:
                    if self.method!='dqn':
                        self.policy.load(file_path)
                        self.policy.eval()
                    else:   # dqn 使用的是Q网络
                        self.online_critic.load(file_path)
                        self.online_critic.eval()
                print("======Evaluating", j*100, "th steps model======")
                self.steps = j*100
                #self.episodes = j*10
                threads = []
                self.try_return_list.clear()
                self.try_success_rate_list.clear()
                self.try_lc_list.clear()
                num_of_eval = 16
                num_of_thread = 4
                num_of_eval_each_thread = int(
                    np.floor(num_of_eval/num_of_thread))
                try_return_q = multiprocessing.Queue()
                try_suc_rate_q = multiprocessing.Queue()
                try_lc = multiprocessing.Queue()
                n = 0
                for i in range(num_of_thread):
                    if i < (num_of_thread-1):
                        try_num_thread = num_of_eval_each_thread
                        num_of_eval = num_of_eval - num_of_eval_each_thread
                    else:
                        try_num_thread = num_of_eval
                    thread = multiprocessing.Process(
                        target=self.evaluate_one_thread, args=(try_num_thread,try_return_q,try_suc_rate_q,try_lc,n,))
                    n+=num_of_eval_each_thread
                    threads.append(thread)
                for i in range(num_of_thread):
                    threads[i].start()
                for i in range(num_of_thread):
                    threads[i].join()
                for i in range(num_of_thread):
                    res = try_return_q.get()
                    self.try_return_list.append(res)
                    rate = try_suc_rate_q.get()
                    self.try_success_rate_list.append(rate)
                    lc = try_lc.get()
                    self.try_lc_list.append(lc)
                self.try_return_list = [
                    item for sublist in self.try_return_list for item in sublist]
                self.try_success_rate_list = [
                    item for sublist in self.try_success_rate_list for item in sublist]
                self.try_lc_list = [item for sublist in self.try_lc_list for item in sublist]
                mean_lc = sum(self.try_lc_list)/len(self.try_lc_list)
                success_rate = sum(self.try_success_rate_list)/len(self.try_success_rate_list)
                mean_return = sum(self.try_return_list) / \
                    len(self.try_return_list)
                max_return = max(self.try_return_list)
                min_return = min(self.try_return_list)
                std_return = np.std(self.try_return_list)/np.sqrt(len(self.try_return_list))
                with open(root_path+'eval_data.txt', 'a') as file:
                    file.write(' '.join(map(str, self.try_return_list)) + '\n')
                with open(root_path+'eval_suc_rate.txt', 'a') as file:
                    file.write(' '.join(map(str, self.try_success_rate_list)) + '\n')
                with open(root_path+'eval_lc.txt', 'a') as file:
                    file.write(' '.join(map(str, self.try_lc_list)) + '\n')
                #sys.stdout = io.StringIO()
                print("最大Return为：", max_return)
                print("平均Return为：", mean_return)
                print("最小Return为：", min_return)
                print("Return Std为：",std_return)
                print("成功率为：",success_rate)
                print("平均换道次数为：",mean_lc)
                lc_list.append(mean_lc)
                success_rate_list.append(success_rate)
                return_list.append(mean_return)
                max_return_list.append(max_return)
                min_return_list.append(min_return)
                std_return_list.append(std_return)
                plot_offline_rewards(
                    return_list, max_return_list, min_return_list,std_return_list, save=True,figure_num=1)
                time.sleep(0.1)
                plot_success_rate(success_rate_list,save=True,figure_num2 =2)
                time.sleep(0.1)
                plot_lc(lc_list,save=True,figure_num =3)
                time.sleep(0.1)
                j = j+1
            except Exception:
                print("Waiting for the ", j*100, "th steps pth")
                time.sleep(5)


        plot_offline_rewards(return_list, max_return_list,
                             min_return_list, save=True)
        plot_success_rate(success_rate_list,save=True)
        plot_lc(lc_list,save=True)

    def evaluate_one_thread(self, try_num_thread,try_return_q,try_suc_rate_q,try_lc,seed):
        #sys.stdout = io.StringIO()
        try_num = 0
        start_time = time.time()
        num_steps = 0
        num_episodes = 0
        max_return = -10000
        min_return = 10000
        try_return_list = []
        try_suc_rate_list = []
        try_lc_list = []
        while try_num < try_num_thread:    
            torch.manual_seed(seed)
            one_thread_env = gym.make("myenv",  render_mode='rgb_array')
            #init_speed = self.ger_uniform_num(self.lower,self.lower+4)
            # mean = 0
            # std = 0.1
            # length = 300
            np.random.seed(seed)
            speed_array = random_select_spd(self.spd_sv,300)
            one_thread_env.set_ref_speed2(speed_array,10)
            np.random.seed(seed)
            one_thread_env.configure(
            {
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "duration": 500,
                "screen_width": 1000,
                "screen_height": 200,
                "show_trajectories": False,
                "render_agent": True,
                "offscreen_rendering": False,
                "initial_lane_id": np.random.choice([0, 1, 2]),
                "vehicles_density": self.ger_uniform_num(1.0, 1.4),
                "ego_spacing":1.5
            })
            if (self.RENDER):
                one_thread_env.config['offscreen_rendering'] = False
            try_num += 1
            state = one_thread_env.reset(seed = seed)
            seed+=1
            #state = self.make_state0(one_thread_env,state[0])
            state = self.make_state(one_thread_env,state[0])
            episode_steps = 0
            episode_return = 0.0
            done, truncate = False, False
            last_current_lane = one_thread_env.config['initial_lane_id']
            stp = 0
            eposide_error = 0
            count = 0
            step_reward = 0
            lc = 0
            while (not done and not truncate):
                stp += 1
                if (self.RENDER == True):
                    one_thread_env.render()
                if count % round(one_thread_env.config['policy_frequency']/self.policy_frequency) == 0:
                    eco_seq = one_thread_env.get_eco_seq()
                    #spd_seq = normalize_speed(one_thread_env.get_speed_seq(100),self.upper,self.lower)
                    if self.method=='sacd' or self.method=='ac':
                        action_d = self.exploit(state,eco_seq)
                        #action_d = self.exploit(state,spd_seq)
                    elif self.method=='dqn':
                        #np.random.seed()
                        action_d = self.dqn_exploit(state,eco_seq)
                        #action_d = self.dqn_exploit(state,spd_seq)
                        #np.random.seed(seed)
                    current_lane = np.clip(round(one_thread_env.vehicle.position[1]/4), 0, 2)
                    if current_lane>0 and action_d == 0:
                        lc+=1
                    elif current_lane<2 and action_d == 2:
                        lc+=1
                action_g, reward_no_solution, no_solution_done,delta_t,cert = self.get_MPC_actions(
                    one_thread_env,action_d, train=True, horizon=10, dt=0.4)
                next_state, reward, done, truncate, info = one_thread_env.step(action_g)
                reward += reward_no_solution
                done = done or no_solution_done
                step_reward = reward
                #next_state = self.make_state0(one_thread_env,np.array(next_state))
                next_state = self.make_state(one_thread_env,np.array(next_state))
                state = next_state
                if count % round(one_thread_env.config['policy_frequency']/self.policy_frequency) == 0 or done or truncate:
                    num_steps += 1
                    episode_steps += 1
                    episode_return += step_reward
                    step_reward = 0
                error = abs(info['speed']-one_thread_env.get_ref_speed()[0])
                eposide_error += error
                count += 1
                if done or truncate:
                    print("回合Return为：", episode_return)
                    try_return_list.append(episode_return)
                    try_lc_list.append(lc)
                    if done:
                        try_suc_rate_list.append(0)
                    elif truncate:
                        try_suc_rate_list.append(1)
            num_episodes += 1
            # eposide_error_array = np.append(eposide_error_array,eposide_error/stp)
        try_return_q.put(try_return_list)
        try_suc_rate_q.put(try_suc_rate_list)
        try_lc.put(try_lc_list)
    
    def MPC_eval(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        eposide_error_array = []
        eposide_return_array = []
        if (self.RENDER):
            self.env.config['offscreen_rendering'] = False
        crash_num = 0
        try_num = 0
        self.seed = 0
        while try_num < 24:
            print("====第 ",try_num+1,"/24"," 次====")
            self.seed+=1
            try_num += 1
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # init_speed = self.ger_uniform_num(self.lower,self.lower+4)
            # mean = 0
            # std = 0.1
            # length = 300
            np.random.seed(self.seed)
            self.env.config['vehicle_density'] = self.ger_uniform_num(1.0,1.4)
            #self.env.config['vehicle_density'] = 1.2
            np.random.seed(self.seed)
            speed_array = random_select_spd(self.spd_sv,300)
            self.env.set_ref_speed2(speed_array,10)
            state = self.env.reset(seed=self.seed)
            state = state[0]
            state = state.reshape([self.state_dim])
            episode_steps = 0 
            episode_return = 0.0
            done, truncate = False, False
            stp = 0
            eposide_error = 0
            count = 0
            last_action_is_change = False
            lane_change_finished = False
            change_flag  = False
            lane_change_count = 0
            target_lane = self.env.config["initial_lane_id"]
            no_solution_flag = False
            while (not done and not truncate):
                stp += 1
                if (self.RENDER == True):
                    self.env.render()
                action_g,last_action_is_change,target_lane,change_flag,lane_change_count,no_solution_flag = MPC_des(self.env,last_action_is_change,count,target_lane,change_flag,lane_change_count,no_solution_flag,horizon = 10,dt = 0.4)
                if not isinstance(action_g,np.ndarray):
                    action_g = np.array([0,0])
                next_state, reward, done, truncate, info = self.env.step(action_g)
                if count % round(self.env.config['policy_frequency']/self.policy_frequency) == 0:
                    episode_steps += 1
                    episode_return += reward
                error = abs(info['speed']-self.env.get_ref_speed()[0])
                eposide_error += error
                count += 1
            num_episodes += 1
            total_return += episode_return

            if done:
                crash_num += 1
            if done or truncate:
                print("回合误差为：", eposide_error/stp)
                print("回合奖励为：",episode_return)
                eposide_error_array = np.append(
                    eposide_error_array, eposide_error/stp)
                eposide_return_array = np.append(
                    eposide_return_array,episode_return)
        end_time = time.time()
        mean_return = total_return / num_episodes
        print("碰撞", crash_num, "/", try_num, "次")
        print("平均误差为：", np.sqrt(np.mean(eposide_error_array**2)))
        print("平均奖励为：", mean_return)
        self.env.close()

    def evaluate(self, process=0, MPC_certi=True):
        self.policy.load("/home/i/sacd/sacd_model/model_SACD_episode_21680.pth")
        self.policy.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        eposide_error_array = []
        eposide_return_array = []
        if (self.RENDER):
            self.env.config['offscreen_rendering'] = False
        crash_num = 0
        try_num = 0
        start_time = time.time()
        while try_num < 10:
            try_num += 1
            state = self.env.reset()
            state = state[0]
            state = state.reshape([self.state_dim])
            last_d = 1
            maintain_stp_max = 5
            maintain_stp = 0
            maintain_flag = False
            episode_steps = 0
            episode_return = 0.0
            done, truncate = False, False
            last_current_lane = self.env.config['initial_lane_id']
            stp = 0
            eposide_error = 0
            count = 0
            last_d = self.env.config['initial_lane_id']
            while (not done and not truncate):
                stp += 1
                if (self.RENDER == True):
                    self.env.render()
                if count % round(self.env.config['policy_frequency']/self.policy_frequency) == 0:
                    eco_seq = self.env.get_eco_seq()
                    spd_seq = normalize_speed(self.env.get_speed_seq(100),self.upper,self.lower)
                    if self.method=='sacd' or self.method=='ac':
                        #action_d = self.exploit(state,eco_seq)
                        action_d = self.exploit(state,spd_seq)
                    elif self.method=='dqn':
                        #action_d = self.dqn_exploit(state,eco_seq)
                        action_d = self.dqn_exploit(state,spd_seq)
                # if not process:
                #     print(action_d)
                # 保持不变
                # if(last_d !=action_d and maintain_flag==False):
                #     maintain_flag = True
                #     maintain_stp+=1
                # elif(maintain_flag and maintain_stp<maintain_stp_max):
                #     action_d = last_d
                #     maintain_stp+=1
                # elif(maintain_flag and maintain_stp>=maintain_stp_max):
                #     maintain_flag = False
                #     maintain_stp = 0
                # 换道完成，强制直行
                current_lane = np.clip(
                    round(self.env.vehicle.position[1]/4), 0, 2)
                if (last_current_lane != current_lane):
                    action_d = 1
                last_current_lane = current_lane
                action_g, reward_no_solution, no_solution_done,delta_t,cert = self.get_MPC_actions(
                    self.env,action_d, train=False, horizon=10, eval_MPC_protect=MPC_certi,dt=0.4)
                next_state, reward, done, truncate, info = self.env.step(
                    action_g)
                last_d = action_d
                reward += reward_no_solution
                if process:
                    done = done or no_solution_done
                next_state = np.array(next_state).reshape([self.state_dim])
                num_steps += 1
                if count % round(self.env.config['policy_frequency']/self.policy_frequency) == 0:
                    episode_steps += 1
                    episode_return += reward
                state = next_state
                error = abs(info['speed']-self.env.get_ref_speed()[0])
                eposide_error += error
                count += 1
            num_episodes += 1
            total_return += episode_return

            if done:
                crash_num += 1
            if process:
                self.episode_rewards.append(episode_return)
                break
            if done or truncate:
                print("回合误差为：", eposide_error/stp)
                print("回合奖励为：",episode_return)
                eposide_error_array = np.append(
                    eposide_error_array, eposide_error/stp)
                eposide_return_array = np.append(
                    eposide_return_array,episode_return)
        if process:
            return
        end_time = time.time()
        mean_return = total_return / num_episodes
        print("碰撞", crash_num, "/", try_num, "次")
        print("平均误差为：", np.sqrt(np.mean(eposide_error_array**2)))
        print("平均奖励为：", mean_return)
        print("平均计算时间为：", (end_time-start_time)/num_steps)
        self.env.close()
        # if mean_return > self.best_eval_score:
        #     self.best_eval_score = mean_return
        #     self.save_models(os.path.join(self.model_dir, 'best'))
