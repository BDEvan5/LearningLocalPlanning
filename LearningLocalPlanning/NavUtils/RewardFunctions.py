import numpy as np

class DistReward:
    # @staticmethod
    def __call__(self, state, s_prime):        
        reward = s_prime['target'][1] - state['target'][1]

        return reward

class SafeReward:
    # @staticmethod
    def __call__(self, state, s_prime):        
        reward = s_prime['target'][1] - state['target'][1]
        if min(s_prime['scan']) < 0.1:
            reward -= 0.2 
        if min(s_prime['scan'][4:7]) < 0.2:
            reward -= 0.2

        return reward

class CthReward:
    def __init__(self, b_ct, b_h):
        self.b_ct = b_ct 
        self.b_h = b_h

    def __call__(self, state, s_prime):        
        # on assumuption of forest with middle @1 and heading =straight 
        pos_x = s_prime['state'][0] 
        reward_ct = abs(1 - pos_x) * self.b_ct 
        reward_h = np.cos(s_prime['state'][2]) * self.b_h

        reward = reward_h - reward_ct

        return reward


