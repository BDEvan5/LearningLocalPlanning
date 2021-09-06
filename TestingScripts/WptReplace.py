from GeneralTestTrain import *
from LearningLocalPlanning.NavAgents.WptReplace import WptReplaceTrain


map_name = "forest2"
n = 1
wpt_rep_name = f"WptRepforest_{n}"
repeat_name = f"WptRepRepeat"

train_n = 200000
test_n = 100


def train_wpt_replace():

    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = WptReplaceTrain(wpt_rep_name, map_name, sim_conf, h_size=200)
    

    train_vehicle(env, vehicle, train_n, False)




if __name__ == "__main__":
    train_wpt_replace()

