from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTest, SerialVehicleTrain

from LearningLocalPlanning.Simulator.ForestSim import ForestSim

import numpy as np
import yaml
from argparse import Namespace


map_name = "forest2"
n = 1
serial_name = f"Serialforest_{n}"


train_n = 200000
test_n = 100


from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace
from ResultsTest import TestVehicles


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



def train_vehicle(env, vehicle, steps):
    done = False
    state = env.reset()

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
            state = env.reset()

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")




"""
Training Functions
"""
def train_serial():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTrain(serial_name, map_name, sim_conf, h_size=200)

    train_vehicle(env, vehicle, train_n)



"""Test Functions"""
def test_serial():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(serial_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False)




def train_repeatability():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = SerialVehicleTrain(train_name, map_name, sim_conf, load=False)

        train_vehicle(env, vehicle, train_n)


def test_repeat():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name, sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, n_test, False)




if __name__ == "__main__":
    # train_serial()
    test_serial()

