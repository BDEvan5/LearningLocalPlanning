
from LearningLocalPlanning.NavAgents.AgentNav import NavTestVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTest 
from LearningLocalPlanning.NavAgents.Oracle import Oracle
from LearningLocalPlanning.NavAgents.FollowTheGap import ForestFGM
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTest

from LearningLocalPlanning.Simulator.ForestSim import ForestSim

import numpy as np
import yaml
from argparse import Namespace


map_name = "forest2"
train_n = 1
nav_name = f"Navforest_{train_n}"
mod_name = f"ModForest_{train_n}"
sap_safe_name = f"SapForest_safe_{train_n}"
sap_cth_name = f"SapForest_cth_{train_n}"
sap_dist_name = f"SapForest_dist_{train_n}"


test_n = 100

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


def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


"""Test Functions"""
def test_nav():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = NavTestVehicle(nav_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False)


def test_follow_the_gap():
    sim_conf = load_conf("", "fgm_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = ForestFGM()

    test_single_vehicle(env, vehicle, True, test_n, add_obs=True, vis=False)


def test_oracle():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = Oracle(sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, True, wait=False)


def test_mod():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = ModVehicleTest(mod_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)

def test_sap_safe():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(sap_safe_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)

def test_sap_dist():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(sap_dist_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)

def test_sap_cth():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(sap_cth_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)


if __name__ == "__main__":
    # test_follow_the_gap()
    # test_oracle()
    # test_mod()
    # test_nav()

    # test_sap_cth()
    test_sap_dist()
    # test_sap_safe()



