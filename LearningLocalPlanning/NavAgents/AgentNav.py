from os import name
import numpy as np 
from numba import njit

import torch
from LearningLocalPlanning.NavUtils.TD3 import TD3, Actor
from LearningLocalPlanning.NavUtils.HistoryStructs import TrainHistory
from LearningLocalPlanning.NavUtils.speed_utils import calculate_speed



class BaseNav:
    def __init__(self, agent_name, sim_conf) -> None:
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        self.distance_scale = 20 # max meters for scaling
        self.range_finder_scale = 5

    def transform_obs(self, obs):
        max_angle = 3.14

        state = obs['state']
        scan = obs['scan'] / self.range_finder_scale
        target = obs['target']

        cur_v = [state[3]/self.max_v]
        cur_d = [state[4]/self.max_steer]
        target_angle = [target[0]/max_angle]
        # target_distance = [target[1]/self.distance_scale]

        nn_obs = np.concatenate([cur_v, cur_d, target_angle, scan])

        return nn_obs

    

class NavTrainVehicle(BaseNav):
    def __init__(self, agent_name, map_name, sim_conf, load=False, h_size=200) -> None:
        BaseNav.__init__(self, agent_name, sim_conf)
        self.path = 'Vehicles/' + agent_name
        state_space = 3 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.try_load(load, h_size, self.path)

        self.t_his = TrainHistory(agent_name, load)

        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_action = None

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)

        nn_action = self.agent.act(nn_obs)
        
        self.state = obs
        self.nn_state = nn_obs
        self.nn_action = nn_action

        steering_angle = self.max_steer * nn_action[0]
        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])

        return self.action

    def calcualte_reward(self, s_prime):
        reward = s_prime['target'][1] - self.state['target'][1]
        # reward = (s_prime[6] - self.state[6]) 
        reward += s_prime['reward']
        
        return reward

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.calcualte_reward(s_prime)

            self.t_his.add_step_data(reward)
            # mem_entry = (self.nn_state, self.nn_action, nn_s_prime, reward, False)
            # self.agent.replay_buffer.add(mem_entry)
            self.agent.replay_buffer.add(self.nn_state, self.nn_action, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        reward = self.calcualte_reward(s_prime)
        nn_s_prime = self.transform_obs(s_prime)
        if self.t_his.ptr % 10 == 0 or True:
            self.t_his.print_update()
            self.agent.save(self.path)
        self.state = None
        self.agent.replay_buffer.add(self.nn_state, self.nn_action, nn_s_prime, reward, False)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)

    def reset_lap(self):
        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_action = None


class NavTestVehicle(BaseNav):
    def __init__(self, agent_name, map_name, sim_conf) -> None:
        BaseNav.__init__(self, agent_name, sim_conf)
        self.path = 'Vehicles/' + agent_name
        self.actor = torch.load(self.path + '/' + agent_name + "_actor.pth")
        
        self.n_beams = 10

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        steering_angle = self.max_steer * nn_action[0]

        speed = calculate_speed(steering_angle)
        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

