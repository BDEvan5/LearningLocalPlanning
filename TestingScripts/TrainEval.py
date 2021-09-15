

from LearningLocalPlanning.NavAgents.SerialAgentPlanner import SerialVehicleTrain, SerialVehicleTest
from LearningLocalPlanning.Simulator.ForestSim import ForestSim
from LearningLocalPlanning.NavUtils.RewardFunctions import *

from GeneralTestTrain import *
import yaml
import time
import csv 

"""
Training Functions
"""
def big_repeat_test(n):
    sim_conf = load_conf("", "test_config")
    test_name = "Repeat"
    agents = range(10)
    sim_conf.test_n = 1000
    lap_time_list = []
    for t in agents:
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{t}_{n}"

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict, lap_times = eval_vehicle_times(env, test_vehicle, sim_conf)
        lap_time_list.append(lap_times)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['test_number'] = n
        config_dict.update(eval_dict)

        # with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            # yaml.dump(config_dict, file)

    directory = f"DataAnalysis/RepeatingTimes_{n}.csv"
    with open(directory, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lap_time_list)

    print(f"Written file: {directory}")

def run_repeatability_tests(n):
    sim_conf = load_conf("", "test_config")
    test_name = "Repeat"
    agents = range(10)
    # agents = [9]
    sim_conf.buffer_n = 1000
    sim_conf.train_n = 20000
    sim_conf.test_n = 100
    for t in agents:
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{t}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        # vehicle.calculate_reward = DistReward()

        train_time = train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['train_time'] = train_time
        config_dict['test_number'] = n
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)


def run_step_tests(n):
    sim_conf = load_conf("", "test_config")
    test_name = "TrainingSteps"
    t_steps = [2000, 4000, 6000, 8000, 10000, 15000, 20000, 40000, 60000]
    
    for t in t_steps:
        sim_conf.train_n = t
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{t}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        # vehicle.calculate_reward = DistReward()

        train_time = train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['train_time'] = train_time
        config_dict['test_number'] = n
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)

def run_beams_tests(n):
    sim_conf = load_conf("", "test_config")
    test_name = "Beams"
    variable_list = [5, 10, 15, 20, 25, 30]
    for variable in variable_list:
        sim_conf.n_beams = variable
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{variable}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        # vehicle.calculate_reward = DistReward()

        train_time = train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['test_number'] = n
        config_dict['train_time'] = train_time
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)

def run_hsize_tests(n):
    sim_conf = load_conf("", "test_config")
    test_name = "SizeH"
    hsizes = [10, 20, 50, 80, 100, 150, 200] 
    sim_conf.buffer_n = 1000
    sim_conf.train_n = 20000
    sim_conf.test_n = 100
    for h in hsizes:
        sim_conf.h_size = h
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{h}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        # vehicle.calculate_reward = DistReward()

        train_time = train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['test_number'] = n
        config_dict['train_time'] = train_time
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)

# reward signals tests
# def run_reward_tests(n):

def compare_rewards(n):
    sim_conf = load_conf("", "test_config")
    test_name = "RewardSignal"
    sim_conf.buffer_n = 1000
    sim_conf.train_n = 20000
    sim_conf.test_n = 100

    rewards = [DistReward(), DistRewardSquare(), DistRewardSqrt(), CthReward(0.004,  0.01), SteeringReward(0.01)]

    for r_signal in rewards:
        env = ForestSim(sim_conf)
        agent_name = f"Sap_{test_name}_{r_signal.name}_{n}"
        training_vehicle = SerialVehicleTrain(agent_name, sim_conf)
        training_vehicle.calculate_reward = r_signal

        train_time = train_vehicle(env, training_vehicle, sim_conf)

        test_vehicle = SerialVehicleTest(agent_name, sim_conf)
        eval_dict = eval_vehicle(env, test_vehicle, sim_conf)

        config_dict = vars(sim_conf)
        config_dict['EvalName'] = test_name 
        config_dict['train_time'] = train_time
        config_dict['test_number'] = n
        config_dict.update(eval_dict)

        with open(f"EvalVehicles/{agent_name}/{agent_name}_record.yaml", 'w') as file:
            yaml.dump(config_dict, file)



if __name__ == "__main__":
    # run_beams_tests(3)
    # run_step_tests(3)
    # run_repeatability_tests(4)
    # run_hsize_tests(2)
    # big_repeat_test(3)
    # big_repeat_test(4)

    compare_rewards(1)
    compare_rewards(2)
    compare_rewards(3)
    compare_rewards(4)

