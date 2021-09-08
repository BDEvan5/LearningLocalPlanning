

from LearningLocalPlanning.NavAgents.AgentNav import NavTrainVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTrain
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTrain
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
from LearningLocalPlanning.NavUtils.RewardFunctions import *

from GeneralTestTrain import train_vehicle, load_conf


map_name = "forest2"
n = 1
# nav_name = f"Navforest_{n}"
# mod_name = f"ModForestDist_{n}"
sap_name_dist = f"SapForestDist_{n}"
sap_name_vel = f"SapForestVel_{n}"
sap_name_steer = f"SapForestSteer_{n}"

train_n = 50000
# train_n = 200


"""
Training Functions
"""
def train_planner(VehicleClass, agent_name, reward):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = VehicleClass(agent_name, map_name, sim_conf, h_size=200)
    vehicle.calculate_reward = reward

    train_vehicle(env, vehicle, train_n)



def train_repeatability(VehicleClass, base_name: str):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, map_name, sim_conf, load=False)

        train_vehicle(env, vehicle, train_n)




if __name__ == "__main__":
    train_planner(SerialVehicleTrain, sap_name_dist, DistReward())
    train_planner(SerialVehicleTrain, sap_name_vel, CthReward(0.004,  0.01))
    train_planner(SerialVehicleTrain, sap_name_steer, SteeringReward(0.01))

    # train_repeatability(ModVehicleTrain, "RepeatMod_forest")
    # train_repeatability(NavTrainVehicle, "RepeatNav_forest")
    # train_repeatability(SerialVehicleTrain, "RepeatSap_forest")

