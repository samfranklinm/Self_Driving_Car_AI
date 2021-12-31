# # AI for Self Driving Car

# # Importing the libraries

# import numpy as np
# import random
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd as autograd
# from torch.autograd import Variable


# # Creating the architecture of the Neural Network

# class Network(nn.Module): # Network inherits from nn.Module (class inheritance)
    
#     def __init__(self, input_size, nb_action): #self, # of input neurons, # of output neurons
#         super(Network, self).__init__()      # to use all the tools in nn.Module on self
#         self.input_size = input_size
#         self.np_action = nb_action
#         self.fc1 = nn.Linear(input_size, 30) # nn.Linear(# of input neurons, 30 hidden nueros (in the hidden layer) -- personal preferences)
#         self.fc2 = nn.Linear(30, nb_action)  # nn.Linear(30 hidden nueros (in the hidden layer, # of output neurons))
        
#     def forward(self, state):
#         x =  F.relu(self.fc1(state))         # x = hidden neurons -> F.relu activates the hidden neurons
#         q_values = self.fc2(x)
#         return q_values
    
# # Implementing Experience Replay

# class ReplayMemory(object):
#     def __init__(self, capacity):            
#         self.capacity = capacity        # capacity = max # of transitions in our memory
#         self.memory = []                # contains past (#) transitions
        
#     def push(self, transition):         # transition = (last state, new state (st+1), last action, last reward) -- a 4-tuple
#         self.memory.append(transition)  # append the transition to the memory
#         if len(self.memory) > self.capacity:    # make sure the memory is always at capacity
#             del self.memory[0]                  # to make room and add another one, delete the oldest transition
        
#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size)) # get random samples from memory list and a fixed size of batch_size and reshape it to the format of 
#                                                                 # if list = ((1,2,3),(4,5,6)), then zip(*list) = ((1,4), (2,3), (5,6))
#         return map(lambda x: Variable(torch.cat(x, 0)), samples) # the lambda function concatenats the  x to the first dimension (torch.cat(x,0))  which contains both tensor and gradient AND the lambda function is applied to samples (x)


# # Implementing Deep Q Learning

# class Dqn():
#     def __init__(self, input_size, nb_action, gamma):
#         self.gamma = gamma
#         self.reward_window = []
#         self.model = Network(input_size, nb_action) # A neural network model that takes an input_size and a nb_action (output neurons)
#         self.memory = ReplayMemory(100000) # ReplayMemory takes a capacity of 100000
#         self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # connect the Adam optimizer to the NN and learning rate of 0.001 (got this through experiment)
#         self.last_state = torch.Tensor(input_size).unsqueeze(0) # create a tensor of size input_size and of a "fake dimension" of 1-D (index 0)
#         self.last_action = 0            # either 0, 20, -20 degrees
#         self.last_reward = 0            # between -1 and 1
        
#     def action_selector(self, state):
#         probs = F.softmax(self.model(Variable(state, volatile = True )) * 10)        # 7 = temperature parameter, higher it is, higher the probability of the selection, 7 is the starting off point
#         action = probs.multinomial(num_samples=1)
#         return action.data[0,0]
    
#     def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
#         outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
#         next_outputs = self.model(batch_next_state).detach().max(1)[0]
#         target = self.gamma*next_outputs + batch_reward
#         td_loss = F.smooth_l1_loss(outputs, target)
#         self.optimizer.zero_grad()
#         td_loss.backward(retain_graph = True)
#         self.optimizer.step()
#     # def learn(self, batch_state, batch_next_state, batch_reward, batch_action):  # parameters = transition
#     #      outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)   # add the chosen action to the fake dimension of its own and then squeeze again to get it back in the tensor format
#     #      next_outputs = self.model(batch_next_state).detach().max(1)[0]        # get the max of q-values of the next state
#     #      target = self.gamma * next_outputs + batch_reward
#     #      td_loss = F.smooth_l1_loss(outputs, target)  # temporal difference - loss and smooth_l1_loss is the best for Dqn (in general)
#     #      self.optimizer.zero_grad()     # reinitialize
#     #      td_loss.backward(retain_graph = True)  # back propagation --- retaining variables/graph will save memory
#     #      self.optimizer.step()          # uses the optimizer to update the weights
         
#     def update(self, reward, new_signal):       # updates when the AI discovers a new state
#         new_state = torch.Tensor(new_signal).float().unsqueeze(0)    # signal = state composed of 5 elements (from map.py (sig1,sig2,sig3,orient,-orient))
#         self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
#         action = self.action_selector(new_state)
#         if len(self.memory.memory) > 100:   # first memory from reply class and second memory is the attribute
#             batch_state, batch_next_state, batch_reward,  batch_action = self.memory.sample(100)
#             self.learn(batch_state, batch_next_state, batch_reward,  batch_action)
#         self.last_action = action
#         self.last_state = new_state
#         self.last_reward = reward
#         self.reward_window.append(reward)
#         if len(self.reward_window) > 1000:
#             del self.reward_window[0]
#         return action

#     def score(self):
#         return sum(self.reward_window)/(len(self.reward_window)+1)

#     def save(self):
#         torch.save({'state_dict': self.model.state_dict(),
#                     'optimizer': self.optimizer.state_dict},
#                     'last_brain.pth')
#     def load(self):
#         if os.path.isfile('last_brain.pth'):
#             print('=> loading checkpoint')
#             checkpoint = torch.load('last_brain.pth')
#             self.model.load_state_dict(checkpoint['state_dict']) # updates the weight,parameters of the model
#             self.model.load_state_dict(checkpoint['optimizer'])  # updates the parameters of the optimizer
#             print('Done.')
#         else:
#             print('No checkpoint found.')


# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
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

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
