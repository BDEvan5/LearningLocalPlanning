from LearningLocalPlanning.NavAgents.AgentSafeSAP import SerialVehicleTrain, SerialVehicleTest
#TODO change name
from LearningLocalPlanning.NavUtils.RewardFunctions import *

from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace



map_name = "forest2"
train_n = 1
sap_dist_name = f"SapForest_sss_{train_n}"
train_n = 200000
# train_n = 1000


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
            env.render(wait=False, name=vehicle.name)
            if wait:
                env.render(wait=True)

        if r == -1:
            crashes += 1
            print(f"({i}) Crashed -> time: {env.steps} ")
            env.render(wait=True, name=vehicle.name)

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
            print(f"Reward: {s_prime['reward']}")
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            # env.render(wait=False, name=vehicle.name)

            vehicle.reset_lap()
            state = env.reset()

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")



def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



def train_sss_dist():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    #TODO: Change name 
    vehicle = SerialVehicleTrain(sap_dist_name, map_name, sim_conf, load=False, h_size=200)
    vehicle.reward_function = SuperSafeReward()
    train_vehicle(env, vehicle, train_n)



def test_sss_dist():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = SerialVehicleTest(sap_dist_name, map_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False, vis=False)


if __name__ == "__main__":
    # train_sss_dist()
    test_sss_dist()

