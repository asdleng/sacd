from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, ActionType
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import scipy.io as scio
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
#from vehicle.behavior import IDMVehicle
dataFile = '/home/i/sacd/ref_spd_data/ele_para.mat'
par = scio.loadmat(dataFile)
par = par['par']
mas = par['mas'][0][0][0][0]
wlr = par['wlr'][0][0][0][0]
fdg = par['fdg'][0][0][0][0]
gav = par['gav'][0][0][0][0]
Cr1 = par['Cr1'][0][0][0][0]
Cr2 = par['Cr2'][0][0][0][0]
rho = par['rho'][0][0][0][0]
ACd = par['ACd'][0][0][0][0]
Mot_maxbr = par['Mot_maxbr'][0][0][0][0]
Mot_Tindx = par['Mot_Tindx'][0][0][0]
Mot_Sindx = par['Mot_Sindx'][0][0][0]
Mot_maxtq = par['Mot_maxtq'][0][0][0]
Mot_map = par['Mot_map'][0][0]
Mot_map = np.nan_to_num(Mot_map, nan=0.0)
Discharge_eff = par['Discharge_eff'][0][0][0][0]
Charge_eff = par['Charge_eff'][0][0][0][0]
Trans_eff = par['Trans_eff'][0][0][0][0]


Observation = np.ndarray


class MyEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    initial_spd = 15
    sur_spd = 10
    start_position = 0
    terminal = 300
    trip_dis = terminal
    speed_array = np.zeros(terminal)
    x_array = np.arange(terminal)
    x_time = np.zeros(terminal)
    TTC = 0
    veh_dis = 0
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
                "normalize": True,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range":[-5,3],
                "steering_range":[-np.pi / 4, np.pi / 4],
                "speed_range": [1, 30]
            },
            "is_IDM": False,  
            "arrive_time_low": 10,  # [s]
            "arrive_time_high": 20, #[s]
            "lanes_count": 3,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "off_road_reward": -10,
            "initial_lane_id": 0,
            "final_lane_id": 0,
            "ego_spacing": 1.5,
            "lane_center_reward": 0.00,
            "road_center_reward":0.00,
            "vehicles_density": 1.0,
            "energy_cost_reward": -0.01,
            "ttc_reward": 0.5,
            "distance_reward": 0.2,
            "collision_reward": -10.0,    # The reward received when colliding with a vehicle.
            "high_speed_reward": 0.0,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "steering_reward": -0.1, # The reward received when steering.
            "lane_change_reward": 0.0,   # The reward received at each lane change action.
            "checkpoint_reward": 0.0, # checkpoint reward.
            "terminal_reward": 10.0, # The reward received when the vehicle finished the task.
            "speed_tracking_reward": 1.0,   # The reward received when vehicle deviate the reference speed.
            "distance_tracking_reward": 1.0,   # The reward received when vehicle deviate the reference speed.
            "reward_speed_range": [10, 20],
            "normalize_reward": False,
            "offroad_terminal": True
        })
        return config
    def get_ref_speed(self) ->np.array:
        if self.speed_array[round(self.trip_dis/2)]==0:
            print("Error, no speed array!")
            return np.array([0,0,0,0,0])
        else:
            p = min([round(self.vehicle.position[0]-self.start_position),self.trip_dis-1])
            ps = self.speed_array[p:p+5]
            if len(ps) is not 5:
                last = ps[-1]
                repeat_count = 5 - len(ps)
                new_array = np.repeat(last, repeat_count)
                ps = np.concatenate((ps, new_array))
            return ps
    def get_t_x(self,t):
        extend = 50
        x_time_prime = self.x_time
        x_array_prime = self.x_array
        if t>x_time_prime[-1]:
            for i in range(extend):
                dt = 1/self.speed_array[-1]
                x_time_prime = np.append(x_time_prime,x_time_prime[-1]+dt)
                x_array_prime = np.append(x_array_prime,x_array_prime[-1]+1)
        x = np.interp(t,x_time_prime,x_array_prime) 
        return x
    def set_ref_speed2(self,spd,sur_spd) -> None:
        self.speed_array = spd
        self.trip_dis = spd.shape[0]
        self.x_array = np.arange(self.trip_dis)
        self.initial_spd = spd[0]
        self.sur_spd = sur_spd
        dt=0
        self.x_time = np.zeros(self.trip_dis)
        for i in range(len(self.x_array)-1):
            dt = 1/self.speed_array[i]
            self.x_time[i+1] = self.x_time[i]+dt
        return
        
    def set_ref_speed(self,file_path) -> None:
        speeds = []
        try:
            with open(file_path, 'r') as file:
                n = 0
                for line in file:
                    if n>=self.trip_dis:
                        break
                    n = n+1
                    # Convert the line to a float and append to the list
                    try:
                        float_number = float(line.strip())
                        speeds.append(float_number)
                    except ValueError:
                        print("Skipped non-float line:", line.strip())
            self.speed_array = np.array(speeds)
            dt=0
            for i in range(len(self.x_array)-1):
                dt = 1/self.speed_array[i]
                self.x_time[i+1] = self.x_time[i]+dt
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    def _reset(self) -> None:
        self.checkpoint_flag = [False,False,False,False]
        self.checkpoint = [300,500,700,900]
        self._create_road()
        self._create_vehicles()


    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:

            f_v_number = round(others/2)
            b_v_number = others - f_v_number
            for _ in range(f_v_number):
                vehicle = other_vehicles_type.create_random(self.road, speed=self.sur_spd, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
            
            # 创建自车：
            if not self.config["is_IDM"]:
                vehicle = Vehicle.create_random(
                    self.road,
                    speed=self.initial_spd,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]
                )
            else:
                vehicle = other_vehicles_type.create_random(
                    self.road,
                    speed=self.initial_spd,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]
                )
                vehicle.randomize_behavior()
            self.start_position = vehicle.position[0]
            self.x_array = np.arange(self.trip_dis)+self.start_position
            self.terminal = self.trip_dis+self.start_position
            if not self.config["is_IDM"]:
                vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
                self.controlled_vehicles.append(vehicle)
            
            self.road.vehicles.append(vehicle)
            self.controlled_id = len(self.road.vehicles)-1    # control vehicle id

            # 创建一个前车，不能太近与自车
            ## Important: 使用这一行，并将b_v_number-1,在训练过程中，以防止上来就撞车
            vehicle = other_vehicles_type.create_random(self.road, speed=self.sur_spd, spacing=self.config["ego_spacing"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
            for _ in range(b_v_number-1):
                vehicle = other_vehicles_type.create_random(self.road, speed=self.sur_spd, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def ttc(self) -> float:
        ttc_list = []
        for v in self.road.vehicles:
            if v is self.vehicle:
                continue
            if abs(v.position[1] - self.vehicle.position[1])<0.2:
                delta = v.position[0] - self.vehicle.position[0]
                ttc = delta / (v.speed-self.vehicle.speed)
                if not np.isnan(ttc) and ttc>0:
                    ttc_list.append(ttc)
        if not ttc_list:
            return -1
        else:    
            return min(ttc_list)
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        #print(rewards["terminal_reward"])
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] ],
                                [0, 1])
        #reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        # speed_tracking_reward:
        speed_tracking_reward = 0
        ref_position = min([round(self.vehicle.position[0]),self.trip_dis-1])
        ref_spd = self.speed_array[ref_position]
        ref_spd_upper_bound = ref_spd + 3
        ref_spd_lower_bound = ref_spd - 3
        if self.vehicle.speed > ref_spd and self.vehicle.speed<ref_spd_upper_bound:
            speed_tracking_reward = utils.lmap(self.vehicle.speed, [ref_spd_upper_bound,ref_spd], [0, 1])
        elif self.vehicle.speed <= ref_spd and self.vehicle.speed>ref_spd_lower_bound:
            speed_tracking_reward = utils.lmap(self.vehicle.speed, [ref_spd_lower_bound,ref_spd], [0, 1])
        #print("speed_tracking_reward:",speed_tracking_reward)
        #print("self.vehicle.speed:",self.vehicle.speed)

        # distance_tracking_reward:
        distance_tracking_reward = 0
        target_x = self.get_t_x(self.time)
        ref_dis_upper_bound = target_x + 10
        ref_dis_lower_bound = target_x - 10
        if self.vehicle.position[0] > target_x and self.vehicle.position[0]<ref_dis_upper_bound:
            distance_tracking_reward = utils.lmap(self.vehicle.position[0], [ref_dis_upper_bound,target_x], [0, 1])
        elif self.vehicle.position[0] <= target_x and self.vehicle.position[0]>ref_dis_lower_bound:
            distance_tracking_reward = utils.lmap(self.vehicle.position[0], [ref_dis_lower_bound,target_x], [0, 1])
        # print("distance_tracking_reward:",distance_tracking_reward)
        # print("self.vehicle.position[0]:",self.vehicle.position[0])
        # print("target_x:",target_x)

        # energy cost reward:
        energy_cost_reward = 0
        energy_cost_reward = self.energy_cost_motor(self.vehicle.speed,self.vehicle.action['acceleration'])*1/self.config["policy_frequency"]

        lane_change_reward = 0
        
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        lat_off = self.vehicle.lane.local_coordinates(self.vehicle.position)[1]
        #print("lat_off:",lat_off)
        if(lat_off>=0):
            lat_off = 2-lat_off
        else:
            lat_off = 2+lat_off
        lane_center_reward = lat_off
        # print("lane_center_reward:",lane_center_reward)
        road_center_reward = 6-abs(self.vehicle.position[1]-4)
        #print("self.vehicle.position[1]", self.vehicle.position[1])
        checkpoint_reward = 0
        break_flag = False
        for i in range(len(self.checkpoint)):
            if break_flag:
                break
            if self.checkpoint_flag[i] is False:
                if self.vehicle.position[0]>self.checkpoint[i]:
                    self.checkpoint_flag[i] = True
                    checkpoint_reward = 1
                    break_flag = True
        # TTC and front distance
        _,sur_veh = self.predict(10,0.1)
        current_lane = np.clip(round(self.vehicle.position[1]/4),0,2)
        ttc = 10
        dis = 50
        for i in range(len(sur_veh)):
            if sur_veh[i][1] == current_lane:
                if sur_veh[i][0] == 0:
                    dis = sur_veh[i][3]
                    if sur_veh[i][2]<10:
                        ttc = sur_veh[i][2]
                    break
        self.TTC = ttc
        self.veh_dis = dis
        if ttc>=5:
            ttc_reward = 1
        elif ttc>=2:
            ttc_reward = utils.lmap(dis, [2,5], [0, 1])
        else:
            ttc_reward = 0
        if dis>=50:
            dis_reward = 1
        elif dis>=10:
            dis_reward = utils.lmap(dis, [10,50], [0, 1])
        else:
            dis_reward = 0

        terminal_reward = 0
        if(self.vehicle.position[0]>self.terminal):
            # if(self.time>self.config["arrive_time_low"] and self.time<self.config["arrive_time_high"]):
            #     if(self.vehicle.lane_index[2] == self.config["final_lane_id"]):
            terminal_t_upper_bound = self.x_time[-1] + 5
            terminal_t_lower_bound = self.x_time[-1] - 5
            if self.time > self.x_time[-1] and self.time<terminal_t_upper_bound:
                terminal_reward = utils.lmap(self.time, [terminal_t_upper_bound,self.x_time[-1]], [0, 1])
            elif self.time <= self.x_time[-1] and self.time>terminal_t_lower_bound:
                terminal_reward = utils.lmap(self.time, [terminal_t_lower_bound,self.x_time[-1]], [0, 1])
            
            #terminal_reward = 1
        
        if action is None:
            action = np.array([0,0])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "steering_reward": float(abs(action[1])),
            #"high_speed_reward": np.clip(scaled_speed, 0, 1),
            "off_road_reward": float(not self.vehicle.on_road),
            "terminal_reward": float(terminal_reward),
            "lane_change_reward": float(lane_change_reward),
            "lane_center_reward": float(lane_center_reward),
            "speed_tracking_reward": float(speed_tracking_reward),
            "distance_tracking_reward": float(distance_tracking_reward),
            #"checkpoint_reward": float(checkpoint_reward),
            #"road_center_reward":float(road_center_reward),
            "ttc_reward": float(ttc_reward),
            "distance_reward": float(dis_reward),
            "energy_cost_reward": float(energy_cost_reward)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        #print(self.vehicle.on_road)
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        #print('current_time',self.time)
        if(self.time>100):
            print("Out of time!")
            return True
        if(self.vehicle.position[0]>self.terminal):
            print("Success!")
            # print('position:',self.vehicle.position[0])
            # print('current_time',self.time)
            return True
        return False
    def predict(self, N, dt):
        """
        predict the surrounding vehicles using constant speed, return an array contains [x,y,heading]
        """
        closed_vehicles = self.road.close_vehicles_to(self.vehicle,
                                                         50,
                                                         count=5,
                                                         see_behind=True,
                                                         sort=True)
        k = -1
        predict_info = []
        surr_vehicles = []
        for vehicle in closed_vehicles:
            k = k+1
            if k == self.controlled_id:
                #print("跳过自车")
                continue
            times = np.arange(N)*dt+dt
            predict_info.append(vehicle.predict_trajectory_constant_speed(times))

            lane_id = np.clip(round(vehicle.position[1]/4),0,2)
            ttc = 1000
            if vehicle.position[0] > self.vehicle.position[0]:
                front_behind = 0
            else:
                front_behind = 1
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            if front_behind == 0:
                dis = vehicle.position[0] - self.vehicle.position[0]
                ttc = dis/ (forward_speed-vehicle.speed)
            if front_behind == 1:
                dis = self.vehicle.position[0] - vehicle.position[0]
                ttc = dis/ (vehicle.speed-forward_speed)
            if ttc<0:
                ttc = 1000
            ttc = np.clip(ttc,0.1,1000)
            surr_vehicles.append([front_behind,lane_id,ttc,dis])
        return predict_info, surr_vehicles

    def energy_cost(self,v,acc):
        m = 1200
        A = 2.5
        umax = 3
        Cd = 0.32
        pa = 1.184
        b0 = 0.1569
        b1 = 0.0245
        b2 = -7.415e-4
        b3 = 5.975e-5
        c0 = 0.0722
        c1 = 0.0968
        c2 = 0.0011
        Fd = 0.1000
        g = 0.81
        miu = 0.0012
        if v < 0.1 or acc<0:
            z = 1
        else:
            z = 0
        f_cruise = b0+b1*v+b2*v**2+b3*v**3
        f_acc = acc*(c0+c1*v+c2*v**2)
        f = (1-z)*(f_cruise + f_acc)+z*Fd
        return f
    def energy_cost_motor(self,v,acc):

        interp_eff = interp2d(Mot_Sindx, Mot_Tindx, Mot_map, kind='linear')
        energy = 0
        mot_spd = v/wlr*fdg
        Ft = mas*acc + 0.5 * ACd * rho * v * v + (Cr1+ Cr2*v) * mas * gav
        mot_tq = Ft*wlr/Trans_eff/fdg;
        mot_tqb = mot_tq.copy()
        interp_maxtq = interp1d(Mot_Sindx,Mot_maxtq,kind='linear')
        if isinstance(mot_spd,  np.ndarray):
            mot_spd[mot_spd<0.0] = 0.0
            mot_spd[mot_spd>Mot_Sindx[-1]] = Mot_Sindx[-1]
        else:
            mot_spd = np.clip(mot_spd,0,Mot_Sindx[-1])
        maxtq = interp_maxtq(mot_spd)
        # 判断是否是数组
        if isinstance(mot_tq, np.ndarray):
            mot_tq[mot_tq>maxtq] = maxtq
            mot_tq[mot_tq<0.0] = 0.0
            mot_tqb[mot_tqb>0.0] = 0.0
            mot_tqb[mot_tqb<-maxtq] = -maxtq
        else:
            mot_tq = np.clip(mot_tq,0,maxtq)
            mot_tqb = np.clip(mot_tqb,-maxtq,0)
        if isinstance(mot_tq, np.ndarray):
            eff_T = interp_eff(mot_spd, mot_tq).diagonal()+1e-12
            eff_B = interp_eff(mot_spd, -mot_tqb).diagonal()+1e-12
        else:
            eff_T = interp_eff(mot_spd, mot_tq)+1e-12
            eff_B = interp_eff(mot_spd, -mot_tqb)+1e-12
        Energy_cost = mot_spd*mot_tq/eff_T/Discharge_eff
        Brake_cost = mot_spd*mot_tqb*eff_B *Charge_eff
        energy = sum(Energy_cost+Brake_cost)
        return energy/1000

    def get_speed_seq(self,length):
        
        if self.speed_array[round(self.trip_dis/2)] == 0:
            print("No speed array for get speed sequence!!!")

        p = round(self.vehicle.position[0]-self.start_position)
        start_p = min([max([0,p]),len(self.speed_array)-1])
        end_p = min([start_p+length,len(self.speed_array)])

        speed_seq = self.speed_array[start_p:end_p]
        while len(speed_seq)<length:
            speed_seq = np.append(speed_seq,speed_seq[-1])
        return speed_seq

    def get_eco_seq(self):
        if self.speed_array[round(self.trip_dis/2)] == 0:
            print("No speed array for get speed sequence!!!")
        spd_seq = self.speed_array
        time_seq = self.x_time
        spd_seq = spd_seq/20
        time_seq = time_seq/30
        return np.reshape(np.concatenate((spd_seq,time_seq),axis=0),[2,len(spd_seq)])
        