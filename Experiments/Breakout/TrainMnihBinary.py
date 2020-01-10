#TODO merge this with the code used throughout the rest of the thesis:
# Challenge: special preprocessing (binarization and combined representation of state and next state) of the replay memory
#            => special preprocessing while running the agent AND in the optimization

# TODO make this code more efficient (time)
#  Ideas: don't use binarization to save memory (long preprocessing time), instead save memory on disk and pull random lines in the optimization

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
from operator import itemgetter
import bitarray

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import logging

import os
import pickle


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


def select_action(policy_net,state,EPS_START,EPS_END,EPS_DECAY,n_actions,device,steps_done):
    """Selects the action with the highest Q value with probability 1-eps_threshold and else a random action
    Args:
        policy_net: the net which determines the action
        state: the current state of the environment
        EPS_START: the initial value of epsilon
        EPS_END: The minimum value of epsilon
        EPS_DECAY: The decay of epsilon
        n_actions: number of possible actions
        device: device for the torch tensors
        steps_done: number of steps done so far to determine the decay
    Returns: The selected action"""
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY) ** steps_done)
    if sample > eps_threshold:
        with torch.no_grad():
            #print(policy_net(state))
            return torch.tensor([[policy_net(state).argmax()]], device=device, dtype=torch.uint8)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.uint8)


def optimize_model(memory,BATCH_SIZE,device,policy_net, Transition, n_actions, target_net,GAMMA,optimizer,double_q_learning=True,
                   gradient_clipping=True,q_value_clipping=False, initial_replay_size=0):
    """Does one update of the optimizer for the policy_net
    Args:
        memory: The reply memory from which transitions are sampled
        BATCH_SIZE: the size of the sample batch
        device: The device for the torch tensors
        Transition: the transition tuple in use
        n_actions: the number of possible actions
        target_net: the current target net
        GAMMA: hyperparameter of the DQN
        optimizer: the optimizer used
        double_q_learning: Whether to use double or "normal" Q-Learning
        gradient_clipping: Whether to use gradient clipping or not
        stack_or_cat: This is a workaround. If the input state has only one dimension the tensors have to be stacked, else they have to be concatenated
    Returns: the updated optimizer and policy_net or None, if the memory is smaller than the batch_size"""
    if len(memory) < BATCH_SIZE or len(memory)<initial_replay_size:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # reassemble states and next states from replay history
    # first reconstruct the replay history
    #TODO if not using binarization, this needs to be adapted
    replay_history = []
    for r in batch.replay_history:
        reconstructed_list = r.tolist()
        reconstructed = torch.tensor(reconstructed_list, device=device,dtype=torch.bool)
        reconstructed = reconstructed.reshape(5, 84, 84)
        replay_history.append(reconstructed)

    state_batch = torch.cat([torch.stack([r[1],r[2],r[3],r[4]],dim=0).unsqueeze(dim=0) for r in replay_history]).float()
    next_state_batch = torch.cat([torch.stack([r[0],r[1],r[2],r[3]],dim=0).unsqueeze(dim=0) for r in replay_history]).float()


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_state_batch)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in next_state_batch
                                         if s is not None])

    action_batch = torch.cat(batch.action).long()
    reward_batch = torch.cat(batch.reward).float()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    if not double_q_learning:
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    else:
        #Double Q Learning
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #We use Double Q Learning, that is the best action is chosen with the policy net, but it's value is calculated with the target net
        next_state_best_actions = torch.zeros([BATCH_SIZE,n_actions], device=device)
        next_state_best_actions[non_final_mask] = policy_net(non_final_next_states)

        best_action_mask = torch.zeros_like(next_state_best_actions)
        best_action_mask[torch.arange(len(next_state_best_actions)), next_state_best_actions.argmax(1)] = 1

        next_state_values[non_final_mask] = target_net(non_final_next_states).masked_select(
            best_action_mask[non_final_mask].byte())

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    if q_value_clipping:
        expected_state_action_values[expected_state_action_values < -10] = -10

    # Compute mse loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()

    loss.backward()
    if gradient_clipping:
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return optimizer, policy_net


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

    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('episode_durations.png')
    '''if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())'''


def save_model(model, directory):
    """Saves a pytorch model"""
    cwd = os.getcwd()
    os.makedirs(directory)
    os.chdir(directory)
    torch.save(model.state_dict(), 'model.pt')
    os.chdir(cwd)
    return


def save_hyperparameters(discount_factor, learning_rate, epsilon_start, epsilon_end, epsilon_decay, replay_memory_size,
                         sample_size, target_update_frequency, episodes, others=''):
    """saves the hyperparameters of the model"""
    with open('hyperparameters.txt', 'w') as file:
        file.write('Discount Factor: ' + str(discount_factor) + '\n')
        file.write('Learning Rate: ' + str(learning_rate) + '\n')
        file.write('Epsilon Start: ' + str(epsilon_start) + '\n')
        file.write('Epsilon End: ' + str(epsilon_end) + '\n')
        file.write('Epsilon Decay: ' + str(epsilon_decay) + '\n')
        file.write('Replay Memory Size: ' + str(replay_memory_size) + '\n')
        file.write('Sample Size: ' + str(sample_size) + '\n')
        file.write('Target Update Frequency: ' + str(target_update_frequency) + '\n')
        file.write('Episodes until model computed: ' + str(episodes) + '\n')
        file.write(others)
    return


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


class TerminalAndFileWriter(object):
    """
    This class allows to write to the Terminal and a file at the same time. It is used for the option save_terminal_output run_scenario_decomposition in runme.py.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self) :
        for f in self.files:
            f.flush()


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize environment
env = gym.make('BreakoutDeterministic-v4')

# define transition tuple
Transition = namedtuple('Transition',
                        ('replay_history', 'action',  'reward'))

# set up result directory
name = 'ResultsMnihBinary'
result_directory = './Results/' + str(name)
os.makedirs(result_directory)
os.chdir(result_directory)

# set up environment
env.reset()

# define hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
# 0.9999977 makes epsilon reach 0.1 after approximately 1000000 frames
EPS_DECAY = 0.9999977
TARGET_UPDATE = 10000
LEARNING_RATE = 0.00025
UPDATE_FREQUENCY = 4
FRAMESKIP = 4
NO_OP_MAX = 30
INITIAL_REPLAY_SIZE = 50000
memory = ReplayMemory(10**6)

env._frameskip = FRAMESKIP
num_episodes = 2*10**5

# write to terminal and file at the same time
terminal_output_file = open('terminal.out', 'w')
# This class allows to write to the Terminal and to any number of files at the same time
sys.stdout = TerminalAndFileWriter(sys.stdout, terminal_output_file)


# Get number of actions from gym action space
n_actions = env.action_space.n

# initialize policy and target net
policy_net = ConvDQN().to(device)
target_net = ConvDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# initialize optimizer
#TODO find out if rmsprop parameters here are set equivalently to the ones used by Mnih et al. in tensorflow
optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE,eps=0.001,alpha=0.95)

# some helper variables
steps_done = 0
episode_durations = []

# save initial model
save_model(policy_net,'initial')
save_hyperparameters(GAMMA, LEARNING_RATE, EPS_START, EPS_END, 'Linear', memory.capacity,BATCH_SIZE, TARGET_UPDATE, 0)
# only save the best n models
number_of_saved_models = 10
saved_models_best_avg = []
saved_model_best_single = []
avg_counter = 0

timer = time.time()

# TODO this function implements binary preprocessing, it can simply be changed to the original preprocessing by removing two lines before the return statement
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
    plt.imshow(state.squeeze()[0].numpy())
    plt.figure()
    plt.imshow(state.squeeze()[1].numpy())
    plt.figure()
    plt.imshow(state.squeeze()[2].numpy())
    plt.figure()
    plt.imshow(state.squeeze()[3].numpy())
    plt.show()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    prev_frame = observation.copy()
    processed_observation = torch.tensor(rgb2gray(observation,prev_frame), device=device,dtype=torch.bool)
    obs_history = []
    obs_history.append(processed_observation)
    accumulated_rewards = 0
    # perform 4-30 no-ops randomly to fill up the history and introduce stochasticity
    no_op = random.randint(4, NO_OP_MAX)
    for i in range(0,no_op):
        # 0 means no operation
        obs, r, done, info = env.step(0)
        # cut off rewards higher than one (according to Mnih)
        r = min(r,1.0)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.bool)
        prev_frame = obs.copy()
        obs_history.insert(0, processed_obs)
        if len(obs_history) > 5:
            obs_history.pop(-1)
        accumulated_rewards += r

    steps=0
    reward = 0.0
    state = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3]],dim=0).unsqueeze(dim=0).float()

    initial_state = state.detach().clone()

    for t in count():
        #if i_episode % 1 == 0:
         #   env.render()
        #if reward > 0:
            #render_state(state)
            #print()
        action = select_action(policy_net,state, EPS_START,EPS_END,EPS_DECAY,n_actions,device,steps_done)
        steps_done += 1
        # execute the action n times according to action repeat
        reward = 0.0
        obs, r, done, info = env.step(action.item())
        # cut off rewards higher than one (according to Mnih)
        r = min(r, 1.0)
        processed_obs = torch.tensor(rgb2gray(obs,prev_frame), device=device,dtype=torch.bool)
        prev_frame = obs.copy()

        reward += r
        obs_history.insert(0,processed_obs)

        accumulated_rewards += reward
        reward = torch.tensor([reward], device=device,dtype=torch.bool)

        # Observe new state
        if not done:
            next_state = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3]],dim=0).unsqueeze(dim=0).float()
            steps += 1
        else:
            next_state = None

        # Store the transition in memory
        # TODO this binarizes the image using bitarray, need to change this if binarization is deleted
        # TODO saving the replay history of the last 5 states is more efficient than saving this state and next state, because they have an overlap of 3 states.
        #  This is implemented but provides a challenge for merging
        replay_history = torch.stack([obs_history[0],obs_history[1],obs_history[2],obs_history[3],obs_history[4]],dim=0)
        as_list = replay_history.flatten().tolist()
        as_bits = bitarray.bitarray(as_list)
        memory.push(as_bits, action, reward)
        #print(len(memory))

        if len(obs_history) > 5:
            obs_history.pop(-1)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network), every 4 selected actions
        if steps_done%UPDATE_FREQUENCY == 0:
            opt = optimize_model(memory,BATCH_SIZE,device,policy_net,Transition,n_actions,target_net,GAMMA,optimizer,double_q_learning=False,initial_replay_size = INITIAL_REPLAY_SIZE)
            if opt is not None:
                optimizer, policy_net = opt
                #print('Q_value: ',policy_net(initial_state))
        # Update the target network, copying all weights and biases in DQN
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if done:
            episode_durations.append(accumulated_rewards)
            # save best single iteration models
            current_perf = accumulated_rewards
            # if less than n models have been saved so far, save it
            if len(saved_model_best_single) < number_of_saved_models:
                saved_model_best_single.append((current_perf, 'best_single_models/trained' + str(i_episode)))
                save_model(policy_net, 'best_single_models/trained' + str(i_episode))
                # sort the models such that the best is the first element, and the worst best the last
                saved_model_best_single.sort(key=itemgetter(0), reverse=True)
            # check if the current mean is higher than the least mean of the best models, if it is not the new model needs not to be saved
            elif current_perf > saved_model_best_single[-1][0]:
                # delete the last entry and the corresponding saved model
                os.remove(saved_model_best_single[-1][1] + '/model.pt')
                os.rmdir(saved_model_best_single[-1][1])
                saved_model_best_single.pop(-1)
                # save the new model and insert it into saved_models
                saved_model_best_single.append((current_perf, 'best_single_models/trained' + str(i_episode)))
                save_model(policy_net, 'best_single_models/trained' + str(i_episode))
                # sort the models such that the best is the first element, and the worst best the last
                saved_model_best_single.sort(key=itemgetter(0), reverse=True)

            # save best average models
            # save the model every other iteration, keep a list of the n best performing models and only save those
            if len(episode_durations) >= 100:
                current_mean = sum(episode_durations[-100:-1]) / len(episode_durations[-100:-1])
                # if less than n models have been saved so far, save it
                if len(saved_models_best_avg) < number_of_saved_models:
                    saved_models_best_avg.append((current_mean, 'best_avg_models/trained' + str(i_episode)))
                    save_model(policy_net, 'best_avg_models/trained' + str(i_episode))
                    # sort the models such that the best is the first element, and the worst best the last
                    saved_models_best_avg.sort(key=itemgetter(0), reverse=True)
                # check if the current mean is higher than the least mean of the best models, if it is not the new model needs not to be saved
                elif current_mean > saved_models_best_avg[-1][0]:
                    # delete the last entry and the corresponding saved model
                    os.remove(saved_models_best_avg[-1][1] + '/model.pt')
                    os.rmdir(saved_models_best_avg[-1][1])
                    saved_models_best_avg.pop(-1)
                    # save the new model and insert it into saved_models
                    saved_models_best_avg.append((current_mean, 'best_avg_models/trained' + str(i_episode)))
                    save_model(policy_net, 'best_avg_models/trained' + str(i_episode))
                    # sort the models such that the best is the first element, and the worst best the last
                    saved_models_best_avg.sort(key=itemgetter(0), reverse=True)
            break
    # save model every x episodes
    if i_episode % 1000 == 0 and i_episode!=0:
        # save replay memory
        #pickle.dump(memory, open("trained"+str(i_episode)+"/replay.p", "wb"))

        #save episode durations
        x_test = np.array(episode_durations)
        np.savez("episode_durations", x_test)
        plot_durations(episode_durations)
        #print 100 episode average
        print('Current Episode: '+str(i_episode))
        print(' Average reward over the last 1000 episodes: '+str(sum(episode_durations[-1000::])/len(episode_durations[-1000::])))
        print('Last 1000 episodes took time: '+str(time.time()-timer))
        print('Current number of frames seen: '+str(steps_done))
        print('Current random action probability: '+str(max(EPS_END, EPS_START * (EPS_DECAY) ** steps_done)))
        print('Best average models: ', saved_models_best_avg)
        print('Best single iteration models: ', saved_model_best_single)
        print('\n\n\n')
        timer = time.time()


print('Best average models: ', saved_models_best_avg)
print('Best single iteration models: ', saved_model_best_single)
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
