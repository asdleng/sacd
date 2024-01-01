import math
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pprint
import casadi as ca
import numpy as np
import time
'''
计算车头时距，不能小于一个值
'''
def tar_head_dis(front_speed):
    head_dis = front_speed*2+2.0
    return head_dis

'''
考虑约束的MPC
'''
def MPC(env,current_state,lane_ind,L, obs,surr_vehicle, n, dt):
    
    T = dt
    N = n            
    current_lane = np.clip(round(env.vehicle.position[1]/4),0,2)
    opti = ca.Opti()
    # control variables, acc and steer
    opt_controls = opti.variable(N, 2)
    acc = opt_controls[:, 0]
    steer = opt_controls[:, 1]
    # states, x, y and heading
    opt_states = opti.variable(N+1, 4)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    v = opt_states[:, 2]
    heading = opt_states[:, 3]

    # parameters
    opt_x0 = opti.parameter(4)
    opt_xs = opti.parameter(4)
    # create model

    def f(x_, u_): 
        beta = ca.arctan(1 / 2 * ca.tan(u_[1]))
        return ca.vertcat(
        x_[2]*ca.cos(x_[3]+beta), x_[2]*ca.sin(x_[3]+beta), u_[0], x_[2]/L*ca.sin(beta))
    def f_np(x_, u_): 
        beta = np.arctan(1 / 2 * np.tan(u_[1]))
        return np.array(
        [x_[2]*np.cos(x_[3]+beta), x_[2]*np.sin(x_[3]+beta), u_[0], x_[2]/L*np.sin(beta)])

    # init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # define the cost function
    # some addition parameters
    Q = np.array([[0.0, 0.0, 0.0,0.0], 
    [0.0, 5.0, 0.0,0.0], 
    [0.0, 0.0, 0.0,0.0], 
    [0.0, 0.0, 0.0, 3.0]])
    R = np.array([[0.1, 0.0], [0.0, 10.0]])
    # cost function
    obj = 0  # cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]
                              ) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    # add constraints to obstacle distance
    closest_vehicle_front_id = -1
    min_dis = 100000
    for i in range(len(obs)):
        if surr_vehicle[i][1] == lane_ind:  # same lane surr vehicle
            if surr_vehicle[i][0] == 1 and surr_vehicle[i][3] <10: # behind
                #if current_lane!=lane_ind: # 如果是换道，才要考虑候车
                for j in range(1,N+1):
                    dx = opt_states[j, 0]-obs[i][0][j-1][0]
                    opti.subject_to((dx - 6)>0)
            if surr_vehicle[i][0] == 0: # front
                if surr_vehicle[i][3] <40:
                    for j in range(1,N+1):
                        dx = obs[i][0][j-1][0] - opt_states[j, 0]
                        opti.subject_to((dx - 6)*(dx + 6)>0)
                ## 第一遍遍历，得到前方最近车
                dis = surr_vehicle[i][3]
                if dis<min_dis:
                    min_dis = dis
                    closest_vehicle_front_id = i

    ttc = 100
    if closest_vehicle_front_id>=0: # 有前车
        if surr_vehicle[closest_vehicle_front_id][3] <100: 
            if surr_vehicle[closest_vehicle_front_id][2]<ttc:
                ttc = surr_vehicle[closest_vehicle_front_id][2]
        if surr_vehicle[closest_vehicle_front_id][3] <100:         
            obj = obj+ 1000 / surr_vehicle[closest_vehicle_front_id][3]
    obj = obj + 800 /  ttc
    for i in range(N):
        obj = obj + 0.01*(v[i] - get_ref_spd(x[i],env))**2
        obj = obj+ 0.01*(x[i] - env.get_t_x(env.time+i*dt))**2
        if closest_vehicle_front_id>=0: # 有前车
            dx = obs[closest_vehicle_front_id][0][i-1][0] - opt_states[i, 0]
            obj = obj + 500/(dx**2)
        
    opti.minimize(obj)

    # boundary and control conditions
    #opti.subject_to(opti.bounded(-2.0, x, 2.0))
    # if lane_ind!=current_lane:
    #     opti.subject_to(opti.bounded(-3.0, y, 11.0))
    # else:
    #     opti.subject_to(opti.bounded(-3.0, y, 11.0))
    #opti.subject_to(opti.bounded(-np.pi/2, heading, np.pi/2))
    opti.subject_to(opti.bounded(0.0, v, 30.0))
    opti.subject_to(opti.bounded(-3.0, acc, 3.0))
    opti.subject_to(opti.bounded(-np.pi/6, steer, np.pi/6))

    opts_setting = {'error_on_fail': False, 'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = np.array([0.0, lane_ind*4, 0.0, 0.0])
    opti.set_value(opt_xs, final_state)
    opti.set_value(opt_x0, current_state)
    opti.set_initial(opt_controls, np.zeros((N, 2)))  # (N, 2)
    initial_states = np.zeros((N+1,4))
    initial_states[0, :] = current_state
    for i in range(N):
        initial_states[i+1, :] = initial_states[i, :] + f_np(initial_states[i, :], np.zeros(2))*T
    opti.set_initial(opt_states, initial_states)  # (N+1, 3)
    try:
      sol = opti.solve()
      u_res = sol.value(opt_controls)
      x_m = sol.value(opt_states)
      obj_value = sol.value(obj)
      return True, u_res, x_m, obj_value
    except RuntimeError:
      return False, np.zeros((N, 2)), initial_states, np.inf
'''
不考虑代价，仅用于验证是否可行
'''
def MPC2(env,current_state,lane_ind, L, obs, surr_vehicle, n, dt):
    
    T = dt
    N = n

    opti = ca.Opti()
    # control variables, acc and steer
    opt_controls = opti.variable(N, 2)
    acc = opt_controls[:, 0]
    steer = opt_controls[:, 1]
    # states, x, y and heading
    opt_states = opti.variable(N+1, 4)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    v = opt_states[:, 2]
    heading = opt_states[:, 3]

    # parameters
    opt_x0 = opti.parameter(4)
    opt_xs = opti.parameter(4)
    # create model

    def f(x_, u_): 
        beta = ca.arctan(1 / 2 * ca.tan(u_[1]))
        return ca.vertcat(
        x_[2]*ca.cos(x_[3]+beta), x_[2]*ca.sin(x_[3]+beta), u_[0], x_[2]/L*ca.sin(beta))
    def f_np(x_, u_): 
        beta = np.arctan(1 / 2 * np.tan(u_[1]))
        return np.array(
        [x_[2]*np.cos(x_[3]+beta), x_[2]*np.sin(x_[3]+beta), u_[0], x_[2]/L*np.sin(beta)])

    # init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)
    # add constraints to obstacle distance
    for i in range(len(obs)):
        if surr_vehicle[i][1] == lane_ind:  # same lane surr vehicle
            if surr_vehicle[i][0] == 1 and surr_vehicle[i][3] <10: # behind
                for j in range(1,N+1):
                    dx = opt_states[j, 0]-obs[i][0][j-1][0]
                    opti.subject_to(dx - 5>0)
            if surr_vehicle[i][0] == 0 and surr_vehicle[i][3] <100: # front
                for j in range(1,N+1):
                    dx = obs[i][0][j-1][0] - opt_states[j, 0]
                    opti.subject_to((dx - 5)*(dx + 5)>0)

    # define the cost function
    # some addition parameters
    Q = np.array([[0.0, 0.0, 0.0,0.0], 
    [0.0, 10.0, 0.0,0.0], 
    [0.0, 0.0, 0.00,0.0], 
    [0.0, 0.0, 0.0, 10.0]])
    R = np.array([[0.0, 0.0], [0.0, 0.0]])
    # cost function
    obj = 0  # cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]
                              ) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    # for i in range(N):
    #     obj = obj + (v[i] - get_ref_spd(x[i],env))**2
    opti.minimize(obj)

    # boundrary and control conditions
    #opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 10.0))
    #opti.subject_to(opti.bounded(-np.pi/4, heading, np.pi/4))
    opti.subject_to(opti.bounded(0.0, v, 30.0))
    opti.subject_to(opti.bounded(-5.0, acc, 3.0))
    opti.subject_to(opti.bounded(-np.pi/4, steer, np.pi/4))

    opts_setting = {'error_on_fail': False, 'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = np.array([0.0, lane_ind*4, 0.0, 0.0])
    opti.set_value(opt_xs, final_state)
    opti.set_value(opt_x0, current_state)
    initial_control = np.zeros((N, 2))
    initial_control[:,0] = -4.9
    opti.set_initial(opt_controls, initial_control)  # (N, 2)
    initial_states = np.zeros((N+1,4))
    initial_states[0, :] = current_state
    for i in range(N):
        initial_states[i+1, :] = initial_states[i, :] + f_np(initial_states[i, :], np.array([-4.9,0]))*T
    opti.set_initial(opt_states, initial_states)  # (N+1, 3)
    try:
      sol = opti.solve()
      u_res = sol.value(opt_controls)
      x_m = sol.value(opt_states)
      obj_value = sol.value(obj)
      return True, u_res, x_m, obj_value
    except RuntimeError:
      return False, np.zeros((N, 2)), initial_states, np.inf
'''不考虑碰撞，训练网络时用
'''
def MPC3(env,current_state,lane_ind, L, obs, surr_vehicle, n, dt):
    
    T = dt
    N = n

    opti = ca.Opti()
    # control variables, acc and steer
    opt_controls = opti.variable(N, 2)
    acc = opt_controls[:, 0]
    steer = opt_controls[:, 1]
    # states, x, y and heading
    opt_states = opti.variable(N+1, 4)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    v = opt_states[:, 2]
    heading = opt_states[:, 3]

    # parameters
    opt_x0 = opti.parameter(4)
    opt_xs = opti.parameter(4)
    # create model

    def f(x_, u_): 
        beta = ca.arctan(1 / 2 * ca.tan(u_[1]))
        return ca.vertcat(
        x_[2]*ca.cos(x_[3]+beta), x_[2]*ca.sin(x_[3]+beta), u_[0], x_[2]/L*ca.sin(beta))
    def f_np(x_, u_): 
        beta = np.arctan(1 / 2 * np.tan(u_[1]))
        return np.array(
        [x_[2]*np.cos(x_[3]+beta), x_[2]*np.sin(x_[3]+beta), u_[0], x_[2]/L*np.sin(beta)])

    # init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)
    # add constraints to obstacle distance
    # for i in range(len(obs)):
    #     if surr_vehicle[i][1] == lane_ind:  # same lane surr vehicle
    #         if surr_vehicle[i][0] == 1 and surr_vehicle[i][3] <10: # behind
    #             for j in range(1,N+1):
    #                 dx = opt_states[j, 0]-obs[i][0][j-1][0]
    #                 opti.subject_to(dx - 5>0)
    #         if surr_vehicle[i][0] == 0 and surr_vehicle[i][3] <40: # front
    #             for j in range(1,N+1):
    #                 dx = obs[i][0][j-1][0] - opt_states[j, 0]
    #                 opti.subject_to((dx - 5)*(dx + 5)>0)

    # define the cost function
    # some addition parameters
    Q = np.array([[0.0, 0.0, 0.0,0.0], 
    [0.0, 5.0, 0.0,0.0], 
    [0.0, 0.0, 0.0,0.0], 
    [0.0, 0.0, 0.0, 3.0]])
    R = np.array([[0.1, 0.0], [0.0, 10.0]])
    # cost function
    obj = 0  # cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]
                              ) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
        obj = obj + 0.01*(v[i] - get_ref_spd(x[i],env))**2
        obj = obj+ 0.01*(x[i] - env.get_t_x(env.time+i*dt))**2
    opti.minimize(obj)

    # boundrary and control conditions
    #opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 10.0))
    #opti.subject_to(opti.bounded(-np.pi/4, heading, np.pi/4))
    opti.subject_to(opti.bounded(0.0, v, 30.0))
    opti.subject_to(opti.bounded(-3.0, acc, 3.0))
    opti.subject_to(opti.bounded(-np.pi/6, steer, np.pi/6))

    opts_setting = {'error_on_fail': False, 'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = np.array([0.0, lane_ind*4, 15, 0.0])
    opti.set_value(opt_xs, final_state)
    opti.set_value(opt_x0, current_state)
    #opti.set_initial(opt_controls, np.zeros((N, 2)))  # (N, 2)
    initial_states = np.zeros((N+1,4))
    initial_states[0, :] = current_state
    for i in range(N):
        initial_states[i+1, :] = initial_states[i, :] + f_np(initial_states[i, :], np.zeros(2))*T
    #opti.set_initial(opt_states, initial_states)  # (N+1, 3)
    try:
      sol = opti.solve()
      u_res = sol.value(opt_controls)
      x_m = sol.value(opt_states)
      obj_value = sol.value(obj)
      return True, u_res, x_m, obj_value
    except RuntimeError:
      return False, np.zeros((N, 2)), initial_states, np.inf


# 计算目标速度
def get_ref_spd(x,env):
    extend = 50
    x_array_prime = env.x_array
    speed_array_prime = env.speed_array
    if env.vehicle.position[0] > x_array_prime[-1]-50:
        for i in range(extend):
            speed_array_prime = np.append(speed_array_prime,speed_array_prime[-1])
            x_array_prime = np.append(x_array_prime,x_array_prime[-1]+1)
    ref_spd = ca.interpolant('C_d_p','linear',[x_array_prime],speed_array_prime)
    return ref_spd(x)
## MPC 决策：
policy_frequency = 1
lane_change_fix_step_num = 3
def MPC_des(env,last_action_is_change,count,target_lane,change_flag,lane_change_count,no_solution_flag,horizon = 10,dt = 0.1):
    ego = env.vehicle
    action = np.array([0,0])
    current_state = np.array([ego.position[0],ego.position[1],ego.speed,ego.heading])
    predict_info,surr_vehicle = env.predict(horizon,dt)
    current_lane = round(ego.position[1]/4)
    current_lane = np.clip(current_lane,0,2)
    candidate_lane = np.array([0,1,2])
    
    status = [0,0,0]
    u_opt = [np.array([0,0]),np.array([0,0]),np.array([0,0])]
    x_opt = [0,0,0]
    obj_value = [0,0,0]
    if(count % round(env.config['policy_frequency']/policy_frequency) == 0):
        if(current_lane==0):
            status[0], u_opt[0], x_opt[0], obj_value[0] = MPC(env,current_state,0,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            status[1], u_opt[1], x_opt[1], obj_value[1] = MPC(env,current_state,1,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            obj_value[2] = np.inf
        elif(current_lane==1):
            status[0], u_opt[0], x_opt[0], obj_value[0] = MPC(env,current_state,0,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            status[1], u_opt[1], x_opt[1], obj_value[1] = MPC(env,current_state,1,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            status[2], u_opt[2], x_opt[2], obj_value[2] = MPC(env,current_state,2,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
        else:
            status[1], u_opt[1], x_opt[1], obj_value[1] = MPC(env,current_state,1,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            status[2], u_opt[2], x_opt[2], obj_value[2] = MPC(env,current_state,2,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            obj_value[0] = np.inf
        best_sol = obj_value.index(min(obj_value))
        if last_action_is_change == True:
            best_sol = current_lane #上次是换道，强制直行
        if best_sol != current_lane:
            last_action_is_change = True
        else:
            last_action_is_change = False
        target_lane = best_sol
        
        # if best_sol != current_lane and change_flag == False:
        #     change_flag = True
        #     
        # if obj_value[target_lane] == np.inf:
        #     print("Wrong initial guess")
        #     best_sol = current_lane
        #     change_flag = False
        #     target_lane = current_lane
        #     no_solution_flag = True
        # if(lane_change_count >= lane_change_fix_step_num):
        #     change_flag = False
        #     lane_change_count = 0
        # if(change_flag):
        #     lane_change_count+=1
        #     action = u_opt[target_lane][0]    
        #     state_pre = x_opt[target_lane][1]
        # else:
        if not best_sol == np.inf:
            action = u_opt[best_sol][0]    
        else:
            action = np.array([0,0])
        #state_pre = x_opt[best_sol][1]
        if no_solution_flag == True:

            status_, u_opt_, x_opt_, obj_value_ = MPC2(env,current_state,best_sol,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            if status_ == False:
                print("Totally no solution!")
                action[0] = -5
            else:
                action = u_opt_[0]
            no_solution_flag = False
    else:
        status_, u_opt_, x_opt_, obj_value_ = MPC(env,current_state,target_lane,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
        action = u_opt_[0]
        if status_ == False:
            status_, u_opt_, x_opt_, obj_value_ = MPC2(env,current_state,target_lane,ego.LENGTH, predict_info,surr_vehicle, horizon, dt)
            action = u_opt_[0]
            if status_ == False:
                print("Totally no solution!")
                action[0] = -5
    
    return action,last_action_is_change,target_lane,change_flag,lane_change_count,no_solution_flag
def main():
    env = gym.make("myenv",  render_mode='rgb_array')
    env.set_ref_speed("/home/i/sacd/data/constant_speed.txt")
    #env.set_ref_speed("/home/i/sacd/data/mountain_curve.txt")
    #env.set_ref_speed("/home/i/sacd/data/speeds.txt")
    env.config["initial_lane_id"] = 1
    #env = NormalizedActions(env)
    env.configure(
        {
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "duration": 500,
            "screen_width": 1000,
            "screen_height": 200,
            "centering_position": [0.5, 0.5],
            "show_trajectories": False,
            "render_agent": False,
            "offscreen_rendering": False
        })
    # 注意，这个才是关键的policy_frequency    

    crash_num = 0
    try_num = 10
    eposide_error_array = np.array([])
    start_time = time.time()
    num_steps = 0
    total_return = 0
    for i in range(try_num):
        eposide_return = 0
        eposide_error = 0
        env.reset()
        done, truncate = False, False
        change_flag  = False
        lane_change_count = 0
        target_lane = env.config["initial_lane_id"]
        no_solution_flag = False
        stp = 0
        count = 0
        while not done and not truncate:
            stp+=1
            num_steps+=1
            env.render()
            
                    
            action,target_lane,change_flag,lane_change_count,no_solution_flag = MPC_des(env,count,target_lane,change_flag,lane_change_count,no_solution_flag)
                

            # vehicle_index, position/heading, xy/timestep,timestep/-
            # action[0]+=np.random.normal(0, 0.1)
            # action[1]+=np.random.normal(0, 0.1)
            next_state, reward, done, truncate, info = env.step(action)
            # dx = env.vehicle.position[0] - state_pre[0]
            # dy = env.vehicle.position[1] - state_pre[1]
            # dv = env.vehicle.speed - state_pre[2]
            # dheading = env.vehicle.heading - state_pre[3]
            #print("delta_x:",round(dx,2),round(dy,2),round(dv,2),round(dheading,2))
            #print("Acc:" ,action[0])
            #print("Speed:" ,info['speed'])
            if(count % round(env.config['policy_frequency']/policy_frequency) == 0):
                eposide_return += reward
            count+=1
            error = abs(info['speed']-env.get_ref_speed()[0])
            eposide_error+=error
            if done:
                crash_num +=1
        total_return+=eposide_return
        print("回合奖励为：",eposide_return)
        print("回合误差为：",eposide_error/stp)
        eposide_error_array = np.append(eposide_error_array,eposide_error/stp)
    end_time = time.time()
    print("碰撞",crash_num,"/",try_num,"次")
    print("平均误差为：",np.sqrt(np.mean(eposide_error_array**2)))
    print("平均奖励为：",total_return/try_num)
    
    print("平均计算时间为：",(end_time-start_time)/num_steps)
    env.close()
if __name__=="__main__":
    main()