# this file implements the ReplayMemory class

from collections import namedtuple
import random
import numpy as np
import os
import torch

# define a transition
Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Replay Memory class"""

    def __init__(self, capacity):
        """capacity is the size of the replay memory"""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self,model):
        """Saves the states of the environment in memory_inputs and the corresponding Q-values in memory_outputs
            Args:
                model: model used to compute the Q-values
            """
        # array for input data
        states = torch.cat([row[0].unsqueeze(dim=0) for row in self.memory])
        memory_inputs = states.detach().numpy()
        # array for q-values ("ground truth")
        memory_outputs = model.forward(states).detach().numpy()
        np.savez("trained/memory_inputs", memory_inputs)
        np.savez("trained/memory_outputs", memory_outputs)


class replay_dataset():
    """dummy dataset consisting of the states and the corresponding actions with the highest Q-value"""
    def __init__(self,path_to_replay,device):
        """Initializes the dataset
        Args:
            device: device for creating torch tensors
        """
        # Read DQN data
        obs = np.load(path_to_replay+"/memory_inputs.npz")
        obs = obs.f.arr_0
        self.observations = torch.from_numpy(obs).type('torch.FloatTensor').to(device)
        # remove dimension of shape 1
        self.observations = self.observations.squeeze()

        q_values = np.load(path_to_replay+"/memory_outputs.npz")
        self.q_values = torch.from_numpy(q_values.f.arr_0).type('torch.FloatTensor').to(device)
        actions = []
        for i in range(0,self.q_values.shape[0]):
            actions.append(torch.argmax(self.q_values[i]))
        self.actions = torch.tensor(actions)

    def next_batch(self, batchsize):
        # Return batch of state-action samples for training
        idx = np.random.choice(self.observations.shape[0], size=batchsize, replace=False)
        return [self.observations[idx], self.actions[idx]]
