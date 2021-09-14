
from LearningLocalPlanning.NavAgents.AgentNav import NavTestVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTest 
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTest
from LearningLocalPlanning.NavAgents.AgentEndToEnd import EndVehicleTest
from LearningLocalPlanning.NavAgents.Oracle import Oracle
from LearningLocalPlanning.NavAgents.FollowTheGap import ForestFGM

from LearningLocalPlanning.Simulator.ForestSim import ForestSim

import numpy as np
import yaml
from argparse import Namespace

from GeneralTestTrain import test_single_vehicle, load_conf, TestVehicles


map_name = "forest2"
train_n = 2
# train_n = "test"
nav_name = f"Navforest_{train_n}"
mod_name = f"ModForest_{train_n}"
sap_name = f"SapForest_{train_n}"
end_name = f"EndForest_{train_n}"

comparison_name = f"ArchComparison_{train_n}"

test_n = 100



"""Test Functions"""
def test_planner(VehicleClass, vehicle_name):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = VehicleClass(vehicle_name, map_name, sim_conf)

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


def comparison_test():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, comparison_name)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(sim_conf)
    test.add_vehicle(vehicle)

    # vehicle = NavTestVehicle(nav_name, map_name, sim_conf)
    # test.add_vehicle(vehicle)
    vehicle = EndVehicleTest(end_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = SerialVehicleTest(sap_name, map_name, sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, test_n, False, wait=False)

def test_repeat(VehicleClass, base_name):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, base_name + "_eval1")

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, map_name, sim_conf)

        test.add_vehicle(vehicle)
    test.run_eval(env, test_n, False)



if __name__ == "__main__":
    # test_follow_the_gap()
    # test_oracle()
    # test_planner(ModVehicleTest, mod_name)
    # test_planner(NavTestVehicle, nav_name)
    # test_planner(SerialVehicleTest, sap_name)
    # test_planner(EndVehicleTest, end_name)

    # test_repeat(NavTestVehicle, "RepeatNav_forest")
    test_repeat(EndVehicleTest, "RepeatEnd_forest")
    test_repeat(ModVehicleTest, "RepeatMod_forest")
    test_repeat(SerialVehicleTest, "RepeatSap_forest")

    # comparison_test()



