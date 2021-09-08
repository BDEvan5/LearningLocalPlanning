

from LearningLocalPlanning.NavAgents.AgentNav import NavTrainVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTrain
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTrain
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace

from GeneralTestTrain import train_vehicle, load_conf


map_name = "forest2"
n = 2
nav_name = f"Navforest_{n}"
mod_name = f"ModForest_{n}"
sap_name = f"SapForest_{n}"

train_n = 20000
# train_n = 200


"""
Training Functions
"""
def train_planner(VehicleClass, agent_name):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = VehicleClass(agent_name, map_name, sim_conf, h_size=200)

    train_vehicle(env, vehicle, train_n)



def train_repeatability(VehicleClass, base_name: str):
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, map_name, sim_conf, load=False)

        train_vehicle(env, vehicle, train_n)




if __name__ == "__main__":

    # train_planner(ModVehicleTrain, mod_name)
    # train_planner(SerialVehicleTrain, sap_name)
    # train_planner(NavTrainVehicle, nav_name)

    # train_repeatability(ModVehicleTrain, "RepeatMod_forest")
    # train_repeatability(NavTrainVehicle, "RepeatNav_forest")
    train_repeatability(SerialVehicleTrain, "RepeatSap_forest")
