from collections import namedtuple
import numpy as np 
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import toy_auto_race.Utils.LibFunctions as lib

MEMORY_SIZE = 100000


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class BufferIL(object):
    def __init__(self, max_size=1000000):     
        #TODO: change from list to array
        self.state_dim = 14 
        self.act_dim = 1 
        self.max_size = max_size

        self.states = np.zeros((max_size, self.state_dim))
        self.actions = np.zeros((max_size, self.act_dim))
        
        self.ptr = 0

    def add(self, state, action):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action

        self.ptr += 1 

        if self.ptr == self.max_size-1: self.ptr=0 

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states = np.zeros((batch_size, self.state_dim))
        actions = np.zeros((batch_size, self.act_dim))

        for i, j in enumerate(ind):
            states[i] = self.states[j]
            actions[i] = self.actions[j]

        return states, actions 

    def size(self):
        return self.ptr

    def load_data(self, name):
        filename = "ImitationData/" + name + ".npy"
        self.storage = np.load(filename)

        print(f"Data loaded: type ({type(self.storage)})")

    def save_buffer(self, name):
        filename = "ImitationData/" + name
        np.save(filename, self.storage)


class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action, h_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x


class ImitationNet:
    def __init__(self, name) -> None:
        self.actor = None
        self.name = name
        self.buffer = BufferIL()

        # self.create()

    def save(self, directory="Vehicles"):
        filename = '%s/%s_actor.pth' % (directory, self.name)

        torch.save(self.actor, filename)

    def load(self, directory="Vehicles"):
        filename = '%s/%s_actor.pth' % (directory, self.name)

        self.actor = torch.load(filename)

    def create(self, obs_dim=14, h_size=200):
        self.actor = Actor(obs_dim, 1, 1, h_size)


    def train(self, batches=5000):
        losses = np.zeros(batches)
        batch_size = 100

        loss = nn.MSELoss()
        optimiser = optim.SGD(self.actor.parameters(), lr=0.001)

        for i in range(batches):
            x, u = self.buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)

            optimiser.zero_grad()

            outputs = self.actor(state)
            actor_loss = loss(outputs[:,0], action)
            actor_loss.backward()
            optimiser.step()

            losses[i] = actor_loss

            if i % 500 == 0:
                print(f"Batch: {i}: Loss: {actor_loss}")

                lib.plot(losses, 100)

                self.save()

        return losses 
