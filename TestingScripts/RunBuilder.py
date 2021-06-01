from collections import namedtuple, OrderedDict
from itertools import product



class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

# class RunManager:
#     def __init__(self):
params = OrderedDict(
    vehicle=
    reward = ,
    name = ,
    map_name = ,
)

for run in RunBuilder.get_runs(params):
    vehicle = run.vehicle(run.name, run.map_name)
    train_vehicle(vehicle, run.env, False)

