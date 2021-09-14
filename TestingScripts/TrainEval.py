

from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTrain, SerialVehicleTest
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
from LearningLocalPlanning.NavUtils.RewardFunctions import *

from GeneralTestTrain import train_vehicle, eval_vehicle, load_conf
import yaml

"""
Training Functions
"""
def train_repeatability(VehicleClass, base_name: str):
    sim_conf = load_conf("", "test_config")
    env = ForestSim(sim_conf)

    for i in range(10):
        train_name = base_name + f"_{i}"

        vehicle = VehicleClass(train_name, sim_conf, load=False)

        train_vehicle(env, vehicle)

def run_network_tests():
    hsizes = [50, 100, 200, 400] 


def run_step_tests(n):
    sim_conf = load_conf("", "test_config")
    test_name = "TrainingSteps"
    # t_steps = [20000, 40000, 80000, 120000, 160000, 200000]
    t_steps = [200, 400, 800, 1200, 1600, 2000]
    # t_steps = [2000]
    sim_conf.buffer_n = 10
    for t in t_steps:
        sim_conf.train_n = t
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{t}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        # vehicle.calculate_reward = DistReward()

        train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)





if __name__ == "__main__":
    run_step_tests(1)


