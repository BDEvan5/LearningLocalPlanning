
from LearningLocalPlanning.NavAgents.AgentNav import NavTestVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTest 
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTest
from LearningLocalPlanning.NavAgents.Oracle import Oracle
from LearningLocalPlanning.NavAgents.FollowTheGap import ForestFGM

from LearningLocalPlanning.Simulator.ForestSim import ForestSim

import numpy as np
import yaml
from argparse import Namespace

from GeneralTestTrain import test_single_vehicle, load_conf, TestVehicles


map_name = "forest2"
train_n = 1
nav_name = f"Navforest_{train_n}"
mod_name = f"ModForest_{train_n}"
sap_name = f"SapForest_dist_{train_n}"

comparison_name = f"ArchComparison_{train_n}"

test_n = 10



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

def test_sap():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(sap_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)


def comparison_test():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, comparison_name)

    vehicle = NavTestVehicle(nav_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = SerialVehicleTest(sap_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, test_n, False, wait=False)


if __name__ == "__main__":
    # test_follow_the_gap()
    # test_oracle()
    # test_mod()
    # test_nav()
    # test_sap()

    comparison_test()


