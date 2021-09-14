

from LearningLocalPlanning.NavAgents.AgentNav import NavTrainVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTrain
from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTrain
from LearningLocalPlanning.NavAgents.AgentEndToEnd import EndVehicleTrain
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace

from GeneralTestTrain import train_vehicle, load_conf


# n = "test"
n = 3
nav_name = f"Navforest_{n}"
mod_name = f"ModForest_{n}"
sap_name = f"SapForest_{n}"
end_name = f"EndForest_{n}"




"""
Training Functions
"""
def train_planner(VehicleClass, agent_name):
    sim_conf = load_conf("", "test_config")
    env = ForestSim(sim_conf)
    vehicle = VehicleClass(agent_name, sim_conf)

    train_vehicle(env, vehicle, sim_conf.train_n)



def train_repeatability(VehicleClass, base_name: str):
    sim_conf = load_conf("", "test_config")
    env = ForestSim(sim_conf)

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, sim_conf, load=False)

        train_vehicle(env, vehicle, sim_conf.train_n)




if __name__ == "__main__":

    train_planner(EndVehicleTrain, end_name)
    train_planner(SerialVehicleTrain, sap_name)
    train_planner(ModVehicleTrain, mod_name)

    # train_repeatability(ModVehicleTrain, "RepeatMod_forest")
    # train_repeatability(EndVehicleTrain, "RepeatEnd_forest")
    # train_repeatability(SerialVehicleTrain, "RepeatSap_forest")
