

import yaml   
from argparse import Namespace
import numpy as np
import csv 

def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



class RunBuilder:
    def __init__(self, config_name):
        self.sim_conf = load_conf('', config_name)

    def build_runs(self, params):
        
        


