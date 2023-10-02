import os
import numpy as np
import torch
from torch.optim import Adam

from .base import BaseAgent
from sacd.model import TwinnedQNetwork2, CateoricalPolicy2
from sacd.utils import disable_gradients
from .mpc_new import MPC,MPC2
from .plot import plot_rewards,plot_durations
class SacdAgent2(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=20000, batch_size=128,
                 lr=0.0003, memory_size=10000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=0,
                 update_interval=2, target_update_interval=2000,
                 use_per=False, dueling_net=False, num_eval_steps=10000,
                 max_episode_steps=1000, log_interval=10, eval_interval=100,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)
        self.RENDER = False
        self.state_dim = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]
        self.act_dim = 3
        self.env.set_ref_speed("/home/i/highway/data/speeds.txt")
        self.env.configure(
        {
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "duration": 500,
            "screen_width": 500,
            "screen_height": 200,
            "show_trajectories": False,
            "render_agent": False,
            "offscreen_rendering": True
        })
        if(self.RENDER):
            self.env.config['offscreen_rendering'] = False
        self.env.set_ref_speed("/home/i/highway/data/speeds.txt")
        self.env.config['initial_lane_id'] = 1

        #
        self.episode_rewards = []
        self.episode_durations = []
        # Define networks.
        self.policy = CateoricalPolicy2(
            self.state_dim, self.act_dim
            ).to(self.device)
        self.online_critic = TwinnedQNetwork2(
            self.state_dim, self.act_dim,
            dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork2(
            self.state_dim, self.act_dim,
            dueling_net=dueling_net).to(device=self.device).eval()

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

    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

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
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
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

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
    def get_MPC_actions(self,action_d):
        action_g = np.array([0,0])
        ego = self.env.vehicle
        dt = 0.1
        current_state = np.array([ego.position[0],ego.position[1],ego.speed,ego.heading])
        predict_info,surr_vehicle = self.env.predict(10,dt)
        current_lane = round(ego.position[1]/4)
        current_lane = np.clip(current_lane,0,2)
        target_lane = np.clip((current_lane + action_d-1),0,2)
        status, u_opt, x_opt, obj_value = MPC(current_state,target_lane,ego.LENGTH, predict_info,surr_vehicle, 10, dt)
        reward_no_solution = 0
        no_solution_done = False
        if status ==False:
            reward_no_solution = -0.001
            #print("No Solution from ", current_lane,"to",target_lane)
            status, u_opt, x_opt, obj_value = MPC2(current_state,current_lane,ego.LENGTH, predict_info,surr_vehicle, 10, dt)
            action_g[0] = u_opt[0,0]
            action_g[1] = u_opt[0,1]
            if status == False:
                reward_no_solution = -0.1
                no_solution_done = True
                #print("Totally no Solution!!!")
                action_g[0] = -5
                action_g[1] = 0
        else:
            action_g = u_opt[0,:]
        return action_g, reward_no_solution, no_solution_done
    
    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done, truncate = False, False
        state = self.env.reset()
        state = state[0]
        state = state.reshape([self.state_dim])
        last_d = 1
        maintain_stp_max = 5
        maintain_stp = 0
        maintain_flag = False
        while (not done and not truncate) and episode_steps <= self.max_episode_steps:
            if(self.RENDER==True):
                self.env.render()
            action_d = self.explore(state)
            #print(action_d)
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

            #action_g, reward_no_solution, no_solution_done = self.get_MPC_actions(action_d)
            next_state, reward, done, truncate, _ = self.env.step(action_d)
            # last_d = action_d
            # reward+=reward_no_solution
            #done  = done or no_solution_done
            next_state = np.array(next_state).reshape([self.state_dim])
            
            # Clip reward to [-1.0, 1.0].
            # clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action_d, reward, next_state, (done or truncate))

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            # if self.steps % self.eval_interval == 0:
            #     self.evaluate(1)
                #self.save_models(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)
        self.episode_rewards.append(episode_return)
        self.episode_durations.append(episode_steps/self.env.config['policy_frequency'])

        if self.episodes % self.log_interval == 0:
            plot_rewards(self.episode_rewards)
            plot_durations(self.episode_durations)
            filename = f"sacd_model2/model_SACD_episode_{self.episodes}.pth"
            torch.save(self.policy.state_dict(), filename)
            # self.writer.add_scalar(
            #     'reward/train', self.train_return.get(), self.steps)

        print(f'Episode: {self.episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {episode_return:<5.1f}')
    # def lift_action(action):
    #     action[0] = (action[0]-(-1))/2*10+(-5)
    #     # action[1] = (action[1]-(-np.pi/6))/2*np.pi/3+(-np.pi/6)
    #     return action
    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break
        # save the policy
        torch.save(self.policy.state_dict(), 'sacd2.pth')
        plot_rewards(self.episode_rewards,show_result=True)
        plot_durations(self.episode_durations,show_result=True)


    def evaluate(self,process=0):
        self.policy.load("/home/i/highway/dsac/sacd_model2/model_SACD_episode_50.pth")
        self.policy.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        if(self.RENDER):
            self.test_env.config['offscreen_rendering'] = False
        crash_num = 0
        try_num = 0
        while True:
            try_num+=1
            state = self.test_env.reset()
            state = state[0]
            state = state.reshape([self.state_dim])
            last_d = 1
            maintain_stp_max = 5
            maintain_stp = 0
            maintain_flag = False
            episode_steps = 0
            episode_return = 0.0
            done,truncate = False,False
            while (not done and not truncate) and episode_steps <= self.max_episode_steps:
                if(self.RENDER==True):
                    self.test_env.render()
                action_d = self.exploit(state)
                if not process:
                    print(action_d)
                # 保持不变
                if(last_d !=action_d and maintain_flag==False):
                    maintain_flag = True
                    action_d = last_d
                    maintain_stp+=1
                elif(maintain_flag and maintain_stp<maintain_stp_max):
                    action_d = last_d
                    maintain_stp+=1
                elif(maintain_flag and maintain_stp>=maintain_stp_max):
                    maintain_flag = False
                    maintain_stp = 0

                action_g, reward_no_solution, no_solution_done = self.get_MPC_actions(action_d)
                next_state, reward, done, truncate, _ = self.test_env.step(action_g)
                last_d = action_d
                reward+=reward_no_solution
                if process:
                    done  = done or no_solution_done
                next_state = np.array(next_state).reshape([self.state_dim])
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return
            
            if done:
                crash_num+=1
            if num_steps > self.num_eval_steps:
                break
            if process:
                self.episode_rewards.append(episode_return)
                self.episode_durations.append(episode_steps/self.env.config['policy_frequency'])
                break
        if process:
            return
        mean_return = total_return / num_episodes
        print("碰撞",crash_num,"/",try_num,"次")
        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

