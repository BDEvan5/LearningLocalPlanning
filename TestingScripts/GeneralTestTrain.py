
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace
# from ResultsTest import TestVehicles

import numpy as np
import csv 

def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



"""General test function"""
def test_single_vehicle(env, vehicle, show=False, laps=100, add_obs=True, wait=False, vis=False):
    crashes = 0
    completes = 0
    lap_times = [] 

    state = env.reset(add_obs)
    done, score = False, 0.0
    for i in range(laps):
        try:
            vehicle.plan_forest(env.env_map)
        except AttributeError as e:
            pass
        while not done:
            a = vehicle.plan_act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
            # env.render(False)
        if show:
            # env.history.show_history()
            # vehicle.history.save_nn_output()
            env.render(wait=False, name=vehicle.name)
            if wait:
                env.render(wait=True)

        if r == -1:
            crashes += 1
            print(f"({i}) Crashed -> time: {env.steps} ")
        else:
            completes += 1
            print(f"({i}) Complete -> time: {env.steps}")
            lap_times.append(env.steps)
        if vis:
            vehicle.vis.play_visulisation()
        state = env.reset(add_obs)
        
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times Avg: {np.mean(lap_times)} --> Std: {np.std(lap_times)}")



"""Testing Function"""
class TestData:
    def __init__(self) -> None:
        self.endings = None
        self.crashes = None
        self.completes = None
        self.lap_times = None
        self.lap_times_no_obs = None

        self.names = []
        self.lap_histories = None

        self.N = None

    def init_arrays(self, N, laps):
        self.completes = np.zeros((N))
        self.crashes = np.zeros((N))
        self.lap_times = np.zeros((laps, N))
        self.lap_times_no_obs = np.zeros((N))
        self.endings = np.zeros((laps, N)) #store env reward
        self.lap_times = [[] for i in range(N)]
        self.N = N
 
    def save_txt_results(self):
        test_name = 'Evals/' + self.eval_name + '.txt'
        with open(test_name, 'w') as file_obj:
            file_obj.write(f"\nTesting Complete \n")
            file_obj.write(f"Map name:  \n")
            file_obj.write(f"-----------------------------------------------------\n")
            file_obj.write(f"-----------------------------------------------------\n")
            for i in range(self.N):
                file_obj.write(f"Vehicle: {self.vehicle_list[i].name}\n")
                file_obj.write(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}\n")
                percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
                file_obj.write(f"% Finished = {percent:.2f}\n")
                file_obj.write(f"Avg lap times: {np.mean(self.lap_times[i])}\n")
                file_obj.write(f"No Obs Time: {self.lap_times_no_obs[i]}\n")

                file_obj.write(f"-----------------------------------------------------\n")

    def print_results(self):
        print(f"\nTesting Complete ")
        print(f"-----------------------------------------------------")
        print(f"-----------------------------------------------------")
        for i in range(self.N):
            if len(self.lap_times[i]) == 0:
                self.lap_times[i].append(0)
            print(f"Vehicle: {self.vehicle_list[i].name}")
            print(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}")
            percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
            print(f"% Finished = {percent:.2f}")
            print(f"Avg lap times: {np.mean(self.lap_times[i])}")
            print(f"No Obs Time: {self.lap_times_no_obs[i]}")
            print(f"-----------------------------------------------------")
        
    def save_csv_results(self):
        test_name = 'Evals/'  + self.eval_name + '.csv'

        data = [["#", "Name", "%Complete", "AvgTime", "Std", "NoObs"]]
        for i in range(self.N):
            v_data = [i]
            v_data.append(self.vehicle_list[i].name)
            v_data.append((self.completes[i] / (self.completes[i] + self.crashes[i]) * 100))
            v_data.append(np.mean(self.lap_times[i]))
            v_data.append(np.std(self.lap_times[i]))
            v_data.append(self.lap_times_no_obs[i])
            data.append(v_data)

        with open(test_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)



class TestVehicles(TestData):
    def __init__(self, config, eval_name, env_kwarg='forest') -> None:
        self.config = config
        self.eval_name = eval_name
        self.vehicle_list = []
        self.N = None
        self.env_kwarg = env_kwarg

        TestData.__init__(self)

    def add_vehicle(self, vehicle):
        self.vehicle_list.append(vehicle)

    def run_eval(self, env, laps=100, show=False, wait=False):
        N = self.N = len(self.vehicle_list)
        self.init_arrays(N, laps)

        # No obstacles
        for j in range(N):
            vehicle = self.vehicle_list[j]

            r, steps = self.run_lap(vehicle, env, show, False, wait)
            self.lap_times_no_obs[j] = env.steps

            print(f"#NoObs: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")

        for i in range(laps):
            env.env_map.add_obstacles()
            for j in range(N):
                vehicle = self.vehicle_list[j]

                r, steps = self.run_lap(vehicle, env, show, False, wait)

                print(f"#{i}: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")
                self.endings[i, j] = r
                if r == -1 or r == 0:
                    self.crashes[j] += 1
                else:
                    self.completes[j] += 1
                    self.lap_times[j].append(steps)

        self.print_results()
        self.save_txt_results()
        self.save_csv_results()

    def run_lap(self, vehicle, env, show, add_obs, wait):
        env.scan_sim.reset_n_beams(vehicle.n_beams)
        state = env.reset(add_obs)

        try:
            vehicle.plan_forest(env.env_map)
        except AttributeError as e:
            pass

        done = False
        while not done:
            a = vehicle.plan_act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
            # env.render(False)

        if show:
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            if wait:
                env.render(wait=True, name=vehicle.name)
            else:
                env.render(wait=False, name=vehicle.name)

        return r, env.steps



def train_vehicle(env, vehicle, steps, obstacles=True, n_buffer=1000):
    done = False
    state = env.reset(obstacles)

    print(f"Building Buffer: {n_buffer}")
    for n in range(n_buffer):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)
        state = s_prime
        
        if done:
            vehicle.done_entry(s_prime)

            vehicle.reset_lap()
            state = env.reset(obstacles)

    print(f"Starting Training: {vehicle.name}")
    for n in range(steps):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        # env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            # env.render(wait=False, name=vehicle.name)

            vehicle.reset_lap()
            state = env.reset(obstacles)

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    print(f"Finished Training: {vehicle.name}")

