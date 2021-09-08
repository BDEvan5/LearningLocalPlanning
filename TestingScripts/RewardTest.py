
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTest

from LearningLocalPlanning.Simulator.ForestSim import ForestSim

import numpy as np
import yaml
from argparse import Namespace

from GeneralTestTrain import test_single_vehicle, load_conf, TestVehicles

map_name = "forest2"
n = 1
sap_name_dist = f"SapForestDist_{n}"
sap_name_vel = f"SapForestVel_{n}"
sap_name_steer = f"SapForestSteer_{n}"


comparison_name = f"RewardComparison_{n}"

test_n = 100



"""Test Functions"""
def test_planner(VehicleClass, vehicle_name):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = VehicleClass(vehicle_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False)


def comparison_test():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, comparison_name)

    vehicle = SerialVehicleTest(sap_name_dist, map_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = SerialVehicleTest(sap_name_vel, map_name, sim_conf)
    test.add_vehicle(vehicle)

    vehicle = SerialVehicleTest(sap_name_steer, map_name, sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, test_n, False, wait=False)

def test_repeat(VehicleClass, base_name):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    test = TestVehicles(sim_conf, base_name + "_eval")

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, map_name, sim_conf)

        test.add_vehicle(vehicle)
    test.run_eval(env, test_n, False)



if __name__ == "__main__":

    # test_planner(SerialVehicleTest, sap_name_dist)
    # test_planner(SerialVehicleTest, sap_name_vel)
    # test_planner(SerialVehicleTest, sap_name_steer)

    comparison_test()

