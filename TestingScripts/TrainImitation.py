import yaml 
from argparse import Namespace
import numpy as np

from LearningLocalPlanning.NavAgents.AgentImitation import ImitationTrain, ImitationTest
from LearningLocalPlanning.NavAgents.FollowTheGap import ForestFGM

from LearningLocalPlanning.Simulator.ForestSim import ForestSim


train_n = 1 
d_name = f"dagger_{train_n}"
map_name = "forest2"
data_set_name = f"dataset_{train_n}"


def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


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
        if show:
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


class TrainDAgger:
    def __init__(self, oracle, env, agent):
        self.env = env
        self.oracle = oracle
        self.agent = agent

    def generate_oracle_data_set(self, init_dset):
        state = self.env.reset()
        print(f"Starting Dataset generation: {self.oracle.name}")
        for i in range(init_dset):
            action = self.oracle.plan_act(state)
            self.agent.save_step(state, action)

            s_prime, r, done, _ = self.env.step_plan(action)
            state = s_prime

            if done:
                # self.env.render(wait=False, name=self.oracle.name)
                self.oracle.reset_lap()
                state = self.env.reset()
            if i %1000 == 0:
                print(f"Data generating... {i}")

        self.agent.buffer.save_buffer(data_set_name)

    def run_dagger(self, n_batches):
        for i in range(n_batches):
            print(f"Running Batch: {i}")
            self.collect_dagger_data(10000)
            self.agent.train()

    def collect_dagger_data(self, dset_size=5000):
        crashes = 0
        completes = 0
        state= self.env.reset()
        print(f"Starting Dataset generation: {self.oracle.name}")
        for i in range(dset_size):
            action = self.agent.plan_act(state)
            oracle_action = self.oracle.plan_act(state)
            self.agent.save_step(state, oracle_action)

            s_prime, r, done, _ = self.env.step_plan(action)
            state = s_prime

            if done:
                # self.env.render(wait=False, name=self.agent.name)
                if s_prime['reward'] == 1:
                    completes += 1
                else:
                    crashes += 1
                self.oracle.reset_lap()
                state = self.env.reset()

            if i %1000 == 0:
                print(f"Data generating... {i}")
        print(f"Completes: {completes} --> Crashes: {crashes} --> Percent: {completes*100/(crashes+ completes)}")

def train_dagger():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    oracle = ForestFGM()
    agent = ImitationTrain(d_name, map_name, sim_conf, 10000)

    d_train = TrainDAgger(oracle, env, agent)

    # d_train.generate_oracle_data_set(20000)
    # agent.buffer.save_buffer(data_set_name)

    d_train.agent.buffer.load_data(data_set_name)
    d_train.agent.train(50000) 
    d_train.run_dagger(15)


def test_dagger():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    agent = ImitationTest(d_name, map_name, sim_conf)

    test_single_vehicle(env, agent, True, test_n)

if __name__ == "__main__":
    train_dagger()
    test_dagger()


