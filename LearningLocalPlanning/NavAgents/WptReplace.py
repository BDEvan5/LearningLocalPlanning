import numpy as np

from LearningLocalPlanning.NavUtils.Trajectoy import Trajectory
from LearningLocalPlanning.NavUtils.TD3 import TD3
from LearningLocalPlanning.NavUtils.HistoryStructs import TrainHistory
from matplotlib import pyplot as plt


class WptReplaceBase:
    def __init__(self, agent_name, map_name, sim_conf):
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.range_finder_scale = 5 #TODO: move to config files

        self.trajectory = Trajectory(map_name)
        self.v = 2

        # self.history = ModHistory()

        self.distance_scale = 20 # max meters for scaling


    def transform_obs(self, obs):
        state = obs['state']
        scan = obs['scan']
        scan = np.clip(scan, 0, self.range_finder_scale)
        scan = scan / self.range_finder_scale

        cur_v = [state[3]/self.max_v]
        cur_d = [state[4]/self.max_steer]

        nn_obs = np.concatenate((scan, cur_v, cur_d))

        
        # n_pts = 5
        # step_distance = 0.3
        # pts = np.zeros((n_pts, 2)) 
        # pts[0] = self.trajectory._get_current_waypoint(state[0:2], step_distance)[0:2]
        # for i in range(1, n_pts):
        #     pts[i] = self.trajectory._get_current_waypoint(pts[i-1], step_distance)[0:2]
        
        # # make pts relative
        # x_scale = 1
        # y_scale = 1.5 
        # r_pts = pts -  np.ones((n_pts, 2)) * state[0:2]
        # r_pts[:, 0] = r_pts[:, 0] / x_scale 
        # r_pts[:, 1] = r_pts[:, 1] / y_scale

        # nn_obs = np.concatenate((r_pts.flatten(), cur_v, cur_d))

        return nn_obs 

    def reset_lap(self):
        pass

class WptReplaceTrain(WptReplaceBase):
    def __init__(self, agent_name, map_name, sim_conf, load=False, h_size=200):
        WptReplaceBase.__init__(self, agent_name, map_name, sim_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 2 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        h_size = h_size
        self.agent.try_load(load, h_size, self.path)

        self.obs = None
        self.nn_obs = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(agent_name, load)


    def plan_act(self, obs):
        """
        Plan an action based on the observation
        """
        self.obs = obs
        nn_obs = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)
        nn_act = self.agent.act(nn_obs)

        self.nn_obs = nn_obs 
        self.nn_act = nn_act

        steering_angle = nn_act[0] * self.max_steer
        action = [steering_angle, self.v]

        if np.isnan(steering_angle):
            nn_act = self.agent.act(nn_obs)
            raise ValueError("Steering angle")

        return action

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_act is not None:
            reward = self.calculate_reward(s_prime)

            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_obs, self.nn_act, nn_s_prime, reward, False)

            # plt.figure(1)
            # plt.clf()
            # plt.xlim([-1, 1])
            # pts = self.nn_obs[:-2]
            # pts = np.reshape(pts, (-1, 2))
            # plt.plot(pts[:, 0], pts[:, 1], '+-', markersize=12)
            # plt.plot(self.nn_act, 0, 'x', markersize=18)
            # plt.title(f"Reward: {reward}")
            # plt.pause(0.0001)


    def calculate_reward(self, s_prime):
        reward = s_prime['target'][1] - self.obs['target'][1]
        # reward = (0.5 - np.abs(s_prime['state'][0]-1)) / 10

        return reward

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = s_prime['reward'] + self.calculate_reward(s_prime)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(True)
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(True)
            self.agent.save(self.path)

        self.agent.replay_buffer.add(self.nn_obs, self.nn_act, nn_s_prime, reward, True)
        self.nn_act = None
