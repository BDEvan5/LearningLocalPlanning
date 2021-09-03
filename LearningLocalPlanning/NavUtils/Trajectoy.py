import numpy as np
from numba import njit, jit
import csv 
from LearningLocalPlanning import LibFunctions as lib


class Trajectory:
    def __init__(self, map_name):
        self.waypoints = None
        self.vs = None
        self._load_csv_track(map_name)
        self.n_wpts = len(self.waypoints)

        

    def _load_csv_track(self, map_name):
        track = []
        filename = 'maps/' + map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt 
        #TODO: make this a parameter
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)


