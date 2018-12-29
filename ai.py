# AI for Self Driving Car

# Importing the libraries

import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib as plt

# The architecture of the Neural Network goes here

class Network(nn.Module):
    
    def __init__(self, input_size, nos_action):#input_size => I/P Layer , nos_action => O/P Layer
        super(Network, self).__init__()
        self.nos_action = nos_action
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 35) # Here only one Hidden layer is created with 35 neurons.
        self.fc2 = nn.Linear(30, nos_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
    
# Implementing Experience Replay

def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

