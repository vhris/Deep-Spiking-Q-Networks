# TODO: Loads the binary Mnih network for Breakout, merging with load_agent from the Code folder should be straightforward

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from skimage.transform import resize
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import statistics

model = './../../Results/Breakout-Mnih-Preliminary-DQN/model.pt'

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.ff(x.view(x.size(0), -1)))
        return self.output_layer(x)


class ReplayMemory(object):
    """Replay Memory class"""

    def __init__(self, capacity):
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


def save_replay_memory(replay_memory, model):
    """Saves the states of the environment in x_test and the corresponding Q-values in y_test
    Args:
        result_directory: current working directory
        replay_memory: replay memory used for saving
        model: model used to compute the Q-values
    """
    # array for input data
    states = torch.cat([row[0] for row in replay_memory.memory])
    x_test = states.detach().numpy()
    # array for q-values ("ground truth")
    y_test = model(states).detach().numpy()
    '''for mem in replay_memory.memory:
        x_test = np.append(x_test,mem[0].detach().numpy())
        y_test = np.append(y_test,model(mem[0]).detach().numpy())'''
    # write to numpy files
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    np.savez("bestavgsofar/x_test", x_test)
    np.savez("bestavgsofar/y_test", y_test)


def plot_durations(episode_durations, plot_mean=True,is_ipython=False, display=None, ylim=None,figure_number=1,title='Training...',xlabel='Episode',ylabel='Duration',default_ones=False):
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
                means = torch.cat((torch.ones(99), means))
            else:
                means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.savefig('episode_durations.png')
    '''if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())'''

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize environment
env = gym.make('BreakoutDeterministic-v4')
FRAMESKIP = 4
NO_OP_MAX = 30
env._frameskip = FRAMESKIP

# set up result directory
name = 'Breakout'
#result_directory = './Results/' + str(name)
#os.chdir(result_directory)

# set up environment
#env.reset()


# Get number of actions from gym action space
n_actions = env.action_space.n

# initialize policy and target net
policy_net = ConvDQN().to(device)
policy_net.load_state_dict(torch.load(model,map_location='cpu'))


# some helper variables
steps_done = 0
episode_durations = []

num_episodes = 50
avg_counter = 0

memory = ReplayMemory(2*10**4)
save_replay = False
finished = False


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


def render_state(state):
    plt.figure()
    plt.imshow(state.squeeze()[0].numpy(), cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(state.squeeze()[1].numpy(), cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(state.squeeze()[2].numpy(), cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(state.squeeze()[3].numpy(), cmap='gray', vmin=0, vmax=255)
    plt.show()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    prev_frame = observation.copy()
    processed_observation = torch.tensor(rgb2gray(observation,prev_frame), device=device,dtype=torch.uint8)
    obs_history = []
    obs_history.append(processed_observation)
    accumulated_rewards = 0
    # perform 4-30 no-ops randomly to fill up the history and introduce stochasticity
    no_op = random.randint(4, NO_OP_MAX)
    for i in range(0,no_op):
        # 0 means no operation
        obs, r, done, info = env.step(0)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.uint8)
        prev_frame = obs.copy()
        obs_history.insert(0, processed_obs)
        if len(obs_history) > 5:
            obs_history.pop(-1)
        accumulated_rewards += r

    steps=0
    reward = 0.0
    state = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3]],dim=0).unsqueeze(dim=0).float()

    for t in count():
        if i_episode % 1 == 0:
            env.render()
        #if reward > 0:
         #   render_state(state)
        sample = random.random()
        eps_threshold = 0.05
        if sample > eps_threshold:
            action = torch.argmax(policy_net(state)).item()
        else:
            action = random.randrange(n_actions)
        #print(policy_net(state))
        #print(action)
        steps_done += 1
        # execute the action n times according to action repeat
        reward = 0.0
        #AI Gym alread implicitly repeats frames 4 times, when using BreakOutNoFrameskip-v4
        obs, r, done, info = env.step(action)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.uint8)
        prev_frame = obs.copy()

        reward += r
        obs_history.insert(0,processed_obs)

        accumulated_rewards += reward
        reward = torch.tensor([reward], device=device,dtype=torch.uint8)

        # Observe new state
        if not done:
            next_state = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3]],dim=0).unsqueeze(dim=0).float()
            steps += 1
        else:
            next_state = None

        # Store the transition in memory
        if save_replay:
            memory.push(state, action, next_state, reward)
            print(len(memory))

        if len(obs_history) > 5:
            obs_history.pop(-1)

        # Move to the next state
        state = next_state

        # save replay memory:
        if save_replay:
            if len(memory) >= memory.capacity and not finished:
                save_replay_memory(memory, policy_net)
                print('replay memory saved')
                # break the training loop
                finished = True
                break

        if done:
            episode_durations.append(accumulated_rewards)
            plot_durations(episode_durations)
            break
    if finished:
        break


print('Complete')
print('Mean: ',statistics.mean(episode_durations))
print('Std: ',statistics.stdev(episode_durations))
env.render()
env.close()
plt.ioff()
plt.show()
