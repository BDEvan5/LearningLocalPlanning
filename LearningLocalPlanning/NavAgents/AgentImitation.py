import torch
from LearningLocalPlanning.NavUtils.DAgger import ImitationNet
from LearningLocalPlanning.NavUtils.speed_utils import calculate_speed
from LearningLocalPlanning.NavUtils import pure_pursuit_utils

import numpy as np 
import csv as csv

class StdPP:
    def __init__(self, sim_conf) -> None:
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.max_steer = sim_conf.max_steer

        self.v_gain = 0.5
        self.lookahead = 0.8
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self.aim_pts = []

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.vs[i])
        else:
            return None

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        self.aim_pts.append(lookahead_point[0:2])

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # speed = 4
        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def reset_lap(self):
        self.aim_pts.clear()


def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return np.array(ret)

def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


class BaseSAP(StdPP):
    def __init__(self, agent_name, map_name, sim_conf) -> None:
        super().__init__(sim_conf)
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.range_finder_scale = 5 #TODO: move to config files

        self.distance_scale = 20 # max meters for scaling

        self._load_csv_track(map_name)

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env
            pp_action: [steer, speed] from pure pursuit controller

        Returns:
            nn_obs: observation vector for neural network
        """
        state = obs['state']
        pp_action = super().act_pp(state)
        scan = obs['scan']/ self.range_finder_scale
        target = obs['target']

        cur_v = [state[3]/self.max_v]
        cur_d = [state[4]/self.max_steer]
        target_angle = [target[0]/self.max_steer]
        dr_scale = [pp_action[0]/self.max_steer]

        nn_obs = np.concatenate([cur_v, cur_d, target_angle, dr_scale, scan])

        return nn_obs

    def modify_references(self, nn_action):
        """
        Modifies the reference quantities for the steering.
        Mutliplies the nn_action with the max steering and then sums with the reference

        Args:
            nn_action: action from neural network in range [-1, 1]
            d_ref: steering reference from PP

        Returns:
            d_new: modified steering reference
        """
        d_new = self.max_steer * nn_action[0] 
        d_new = np.clip(d_new, -self.max_steer, self.max_steer)

        return d_new

    def _load_csv_track(self, map_name):
        track = []
        filename = 'maps/' + map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)



class ImitationTrain(ImitationNet, BaseSAP): 
    def __init__(self, agent_name, map_name, sim_conf, train_steps=10000):
        ImitationNet.__init__(self, agent_name)
        BaseSAP.__init__(self, agent_name, map_name, sim_conf)

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

        self.train_steps = train_steps 

    def load_buffer(self, buffer_name):
        self.buffer.load_data(buffer_name)

    def save_step(self, state, action):
        nn_obs = self.transform_obs(state)
        action = action[0] / self.max_steer
        self.buffer.add(nn_obs, action)

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)
        # TODO: add noise in these readings.
        nn_act = self.actor(nn_obs).detach().numpy()
        nn_act += np.random.normal(0, 0.1, size=1)

        steering_angle = self.modify_references(nn_act)
        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])

        return self.action


class ImitationTest(BaseSAP):
    def __init__(self, agent_name, map_name, sim_conf) -> None:
        BaseSAP.__init__(self, agent_name, map_name, sim_conf)

        filename = '%s/%s_actor.pth' % ("Vehicles", self.name)
        self.actor = torch.load(filename)

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_act = self.actor(nn_obs).detach().numpy()

        steering_angle = self.modify_references(nn_act)
        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])

        return self.action

