# TODO this file implements converting the binary Mnih network. Merging it with the rest of the code comes with the following challenges:
#  1. this file comes with its own SQN class, because the one in Code does not support convolutional layers, this file uses a hacky version which allows to use convolutional layers
#     but in the future a nicer version for convolutional layers should be implemented

# TODO this network does not achieve near loss-less conversion as Patel et al. report for 500 time steps. Why does it perform worse?

import gym
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import random
import numpy as np
from skimage.transform import resize
import time
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import logging

import os
import pickle
import statistics

class ConvDQN(nn.Module):
    """The DQN for the original BreakOut problem"""

    def __init__(self):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)
        self.ff = nn.Linear(3136,512)
        self.output_layer = nn.Linear(512,4)
        # self.apply(weights_init_uniform)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.ff(x.view(x.size(0), -1)))
        return self.output_layer(x)

    def forward_return_all(self,x):
        x = x.unsqueeze(dim=0)
        all_neurons_output = []
        x = F.relu(self.conv1(x))
        all_neurons_output.append(x)
        x = F.relu(self.conv2(x))
        all_neurons_output.append((x))
        x = F.relu(self.conv3(x))
        all_neurons_output.append(x)
        x = F.relu(self.ff(x.view(x.size(0), -1)))
        all_neurons_output.append(x)
        x = self.output_layer(x)
        all_neurons_output.append(x)
        return all_neurons_output


def plot_durations(episode_durations, plot_mean=True,is_ipython=False, display=None, ylim=None,figure_number=1,title='Training...',xlabel='Episode',ylabel='Duration',default_ones=False,max_steps=0):
    """Plots the duration length for the episodes of a gym environment"""
    plt.figure(figure_number)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(durations_t.numpy())
    if ylim is not None:
        plt.ylim((0,ylim))

    if plot_mean:
        # Take 100 (or len(epsisodes)) episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            if default_ones:
                means = torch.cat((torch.ones(99)*max_steps, means))
            else:
                means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    '''if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())'''


def plot_mean(episode_durations):
    """plots the average over all episodes"""
    # Plot average
    durations = np.array(episode_durations)
    mean = np.mean(durations)
    plt.plot((0, len(episode_durations)-1), (mean, mean), 'orange')
    plt.title('Mean: ' + str(mean))


class dataset():
    """dummy dataset consisting of the states and the corresponding actions with the highest Q-value"""
    def __init__(self,device, add_bias_as_observation=False, normalization=False):
        """Initializes the dataset
        Args:
            device: device for creating torch tensors
            add_bias_as_observation: If True each observation is appended with 1 (used for directly training a SNN)
            normalization: If True the observations are min-max rescaled with the highest and lowest value from the dataset for each observation
        """
        # Read DQN data
        obs = np.load("x_test.npz")
        if add_bias_as_observation:
            # add bias as an input of 1 to obs
            obs = obs.f.arr_0
            obs = np.c_[obs, np.ones_like(obs[:, 0]).reshape(obs.shape[0], 1)]
        else:
            obs = obs.f.arr_0
        self.observations = torch.from_numpy(obs).type('torch.FloatTensor').to(device)
        # remove dimension of shape 1
        self.observations = self.observations.squeeze()
        if normalization:
            # normalize observations in a way that all values are positive
            # use min max rescaling for negative and pt
            for i in range(0, self.observations.shape[1]):
                high = torch.max(self.observations[:, i])
                low = torch.min(self.observations[:, i])
                # normalize negative and positive values seperately
                self.observations[:, i] = (self.observations[:, i] - low) / (high - low)

        q_values = np.load("y_test.npz")
        self.q_values = torch.from_numpy(q_values.f.arr_0).type('torch.FloatTensor').to(device)
        actions = []
        for i in range(0,self.q_values.shape[0]):
            actions.append(torch.argmax(self.q_values[i]))
        self.actions = torch.tensor(actions)



class SQN():

    def __init__(self,model, weights, bias, normalization, device):
        """Args:
            model: the model that is converted
            weights: a list, where the first element is the weights of the first layer of the model and so on
            bias: as weights, but for the bias
            normalization: if True the dataset is normalized
            device: device which torch is using"""

        # self.alpha = float(np.exp(-time_step / tau_syn))
        # self.beta = float(np.exp(-time_step / tau_mem))
        self.alpha = 0
        self.beta = 1
        self.model_copy = model
        self.time_step = 1e-3

        # Get weights from trained network
        self.weights = weights
        self.bias = bias

        # use training set to find max_act for each neuron
        # better normalization method according to Diehl paper
        training_set = dataset(device, normalization=normalization)
        activations = []
        for l in range(0, len(self.weights)):
            activations.append([])
        max_it = 20000
        current_it = 0
        for obs in training_set.observations:
            current_it += 1
            if current_it > max_it:
                break
            output = model.forward_return_all(obs)
            for x in range(0,len(activations)):
                activations[x].append(torch.max(output[x]))

        previous_factor = 1
        for l in range(len(self.weights)):
            # get the p-percentile of the activation
            pos_inputs = activations[l]
            pos_inputs.sort()
            ppercentile = 0.99
            max_act = pos_inputs[int(ppercentile * (len(pos_inputs) - 1))]
            # get the maximum weight in the layer
            max_wt = torch.max(self.weights[l])
            if self.bias[l] is not None:
                max_bias = torch.max(self.bias[l])
                max_wt = torch.max(max_wt, max_bias)
            scale_factor = max(max_wt, max_act)
            if scale_factor <= 0:
                raise Exception('scale factor smaller than 0, this is unexpected')
            applied_factor = scale_factor / previous_factor
            # rescale weights
            self.weights[l] = self.weights[l] / applied_factor

            # rescale bias
            if self.bias[l] is not None:
                self.bias[l] = self.bias[l] / scale_factor
            previous_factor = scale_factor


        # manual setting of correct weights
        # TODO automatically set the weights, easiest way is to have a layer array in the DQN class which contains all layers
        self.model_copy.conv1.weight.data = self.weights[0]
        self.model_copy.conv1.bias.data = self.bias[0]
        self.model_copy.conv2.weight.data = self.weights[1]
        self.model_copy.conv2.bias.data = self.bias[1]
        self.model_copy.conv3.weight.data = self.weights[2]
        self.model_copy.conv3.bias.data = self.bias[2]
        self.model_copy.ff.weight.data = self.weights[3]
        self.model_copy.ff.bias.data = self.bias[3]
        self.model_copy.output_layer.weight.data = self.weights[4]
        self.model_copy.output_layer.bias.data = self.bias[4]
        self.layers = []
        self.layers.append(self.model_copy.conv1)
        self.layers.append(self.model_copy.conv2)
        self.layers.append(self.model_copy.conv3)
        self.layers.append(self.model_copy.ff)
        self.layers.append(self.model_copy.output_layer)


    # Heaviside step function as the spiking non linearity
    def spike_fn(self, x):
        out = torch.zeros_like(x)
        out[x > 0] = 1.0
        return out

    def simulate(self, batch_size, input_data, input_size, timesteps,device):
        """Simulates the network for the specified number of timesteps

        Args:
            batch_size: Number of inputs
            input_data: tensor of the input data
            input_size: the number of input neurons
            timesteps: simulation length
            device: device which torch is using

        Returns:
             mem_rec: the recorded membrane potentials during the simulation
             spk_rec: the recorded spikes during the simulation
             spk_count: the total number of recorded spikes for each neuron"""

        # reset the array for membrane potential and synaptic variables
        syn = []
        mem = []
        spk_count = []
        spk_count.append(torch.zeros((batch_size, input_size), device=device, dtype=torch.float))
        example_output = torch.zeros(batch_size,4,84,84)
        for l in range(0, len(self.weights)):
            # TODO find out an automatic way to determine the first non convolutional layer where a reshape is necessary
            if l == 3:
                example_output = example_output.view(example_output.size(0), -1)
            # TODO: hacky way to find out correct layer sizes: just apply the layers from the PyTorch network and observe the output size
            example_output = self.layers[l](example_output)
            syn.append(torch.zeros_like(example_output,device=device,dtype=torch.float))
            mem.append(torch.zeros_like(example_output,device=device,dtype=torch.float))
            spk_count.append(torch.zeros_like(example_output,device=device,dtype=torch.float))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(timesteps):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            # constant input current
            input = torch.tensor(input_data, dtype=torch.float)

            #spk_count[0] += input

            # loop over layers
            for l in range(len(self.weights)):
                # define impulse
                # TODO this is a very hacky way to deal with convolutional layers (or any other kind of layers):
                #  We simply use the layer of the original PyTorch network with converted weights to compute the forward pass on the spikes of the last layer
                if l == 0:
                    h = self.layers[l](input)
                # TODO again, find an automatic way to find layers which need reshaping (i.e. first layer after convolution)
                elif l==3:
                    h = self.layers[l](spk_rec[len(spk_rec) - 1][l - 1].view(spk_rec[len(spk_rec) - 1][l - 1].size(0), -1))
                else:
                    h = self.layers[l](spk_rec[len(spk_rec) - 1][l - 1])

                # separate output layer (does not spike):
                if l == len(self.weights) - 1:
                    new_syn = self.alpha * syn[l] + h
                    # reset by subtraction
                    new_mem = self.beta * mem[l] + syn[l]

                    mem[l] = new_mem
                    syn[l] = new_syn

                    mem_rec[len(mem_rec) - 1].append(mem[l])
                    spk_rec[len(spk_rec) - 1].append(new_mem)
                    spk_count[l + 1] = new_mem
                else:
                    # the -1 implies that the threshold is 1
                    mthr = mem[l] - 1
                    out = self.spike_fn(mthr)
                    rst = torch.zeros_like(mem[l])
                    c = (mthr > 0)
                    rst[c] = torch.ones_like(mem[l])[c]

                    new_syn = self.alpha * syn[l] + h
                    # reset by subtraction
                    new_mem = self.beta * mem[l] + syn[l] - rst

                    mem[l] = new_mem
                    syn[l] = new_syn

                    mem_rec[len(mem_rec) - 1].append(mem[l])
                    spk_rec[len(spk_rec) - 1].append(out)
                    spk_count[l + 1] += out


        return mem_rec, spk_rec, spk_count



os.chdir('./../../Results/Breakout-Mnih-Preliminary-DQN')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize environment
env = gym.make('BreakoutDeterministic-v4')

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action','next_state',  'reward'))

# set up environment
env.reset()

#torch.random.manual_seed(0)
# define hyperparameters
FRAMESKIP = 4
NO_OP_MAX = 30

finished = False

env._frameskip = FRAMESKIP
num_episodes = 100

# Get number of actions from gym action space
n_actions = env.action_space.n

# initialize dqn
dqn = ConvDQN()
dqn.load_state_dict(torch.load('model.pt',map_location=device))
dqn_copy = ConvDQN()
dqn_copy.load_state_dict(torch.load('model.pt',map_location=device))

# set up snn
W1 = dqn.conv1.weight
W2 = dqn.conv2.weight
W3 = dqn.conv3.weight
W4 = dqn.ff.weight
W5 = dqn.output_layer.weight
weights = [W1, W2,W3,W4,W5]
b1 = dqn.conv1.bias
b2 = dqn.conv2.bias
b3 = dqn.conv3.bias
b4 = dqn.ff.bias
b5 = dqn.output_layer.bias
bias = [b1, b2,b3,b4,b5]
# Initialization of SNN
snn = SQN(dqn_copy,weights, bias,False,device)

mismatches = 0
correct = 0
# some helper variables
steps_done = 0
episode_durations = []

def rgb2gray(rgb,prev_frame):
    """converts a coloured image to gray"""
    # take the maximum value of this frame and the previous frame:
    image_max = rgb.copy()
    image_max[prev_frame>rgb] = prev_frame[prev_frame>rgb]
    # extract the luminance
    image = image_max[:,:,0]*0.2126+image_max[:,:,1]*0.7152+image_max[:,:,2]*0.0722
    # downsize to 84x84
    image_resized = resize(image,(84,84))
    # binarize the image
    th_bin = 5
    image_resized = image_resized>th_bin
    # return the processed image and the new prev frame
    return image_resized



for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    prev_frame = observation.copy()
    processed_observation = torch.tensor(rgb2gray(observation,prev_frame), device=device,dtype=torch.float32)
    obs_history = []
    obs_history.append(processed_observation)
    accumulated_rewards = 0
    # perform 4-30 no-ops randomly to fill up the history and introduce stochasticity
    no_op = random.randint(4, NO_OP_MAX)
    for i in range(0,no_op):
        # 0 means no operation
        obs, r, done, info = env.step(0)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.float32)
        prev_frame = obs.copy()
        obs_history.insert(0, processed_obs)
        if len(obs_history) > 4:
            obs_history.pop(-1)
        accumulated_rewards += r

    steps=0
    reward = 0.0
    state = torch.stack([obs_history[0], obs_history[1], obs_history[2], obs_history[3]], dim=0).unsqueeze(dim=0).float()

    initial_state = state.detach().clone()
    for t in count():
        if i_episode % 1 == 0:
            env.render()
        #if reward>0.0:
         #   render_state(state)
           # print()

        rand = random.random()
        if rand>0.1:
            q_values = snn.simulate(1,state,28224,500,device)[-1][-1]
            q_values_comp = dqn(state.squeeze(dim=0))
            print(q_values,q_values_comp)
            if torch.argmax(q_values) != torch.argmax(q_values_comp):
                print('mismatch')
                mismatches += 1
            else:
                correct += 1
            if (correct + mismatches) % 100 == 0:
                print('Accuracy after ' + str(correct + mismatches) + ' iterations: ' + str(
                    correct * 100 / (mismatches + correct)) + '%')
            action = torch.argmax(q_values).item()
        else:
            action = random.randint(0,3)
        steps_done += 1

        reward = 0.0

        obs, r, done, info = env.step(action)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.float32)
        prev_frame = obs.copy()

        reward += r
        obs_history.insert(0,processed_obs)

        accumulated_rewards += reward
        reward = torch.tensor([reward], device=device,dtype=torch.float32)

        # Observe new state
        if not done:
            next_state = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3]],dim=0).unsqueeze(dim=0).float()
            steps += 1
        else:
            next_state = None

        if len(obs_history) > 4:
            obs_history.pop(-1)

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(accumulated_rewards)
            plot_durations(episode_durations,plot_mean=False)
            break


print('Mean: ',statistics.mean(episode_durations))
print('Std: ',statistics.stdev(episode_durations))

plot_mean(episode_durations)
print('Accuracy after '+str(correct+mismatches)+' iterations: '+ str(correct*100/(mismatches+correct))+'%')
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
