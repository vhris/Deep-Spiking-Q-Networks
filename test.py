from Code.train_agent import train_agent
from Code.NeuralNetworks import FullyConnected
from Code.load_agent import load_agent
from Code.train_classifier import train_classifier
from Code.SQN import SQN,SuperSpike, weight_conversion
from Code.utils import save_model
from Code.NEST_agent import Nestwork
from Code.PyNN_agent import PyNNAgent
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
import random


def train_cartpole_dqn(spiking):
    """Test training of a DQN on CartPole, if spiking is True a DSQN is used, else a DQN"""
    #CartPole
    env = 'CartPole-v0'

    os.chdir('Results/Test')

    # determine seeds
    #torch_seed = random.randint(0, 1000)
    torch_seed =135
    torch.manual_seed(torch_seed)
    #gym_seed = random.randint(0, 1000)
    gym_seed = 975
    #random_seed = random.randint(0, 1000)
    random_seed = 795
    random.seed(random_seed)
    print('Seeds (torch,gym,random): ', torch_seed, gym_seed, random_seed)

    # initialize policy and target net
    if not spiking:
        architecture = [4,16,16,2]
        policy_net = FullyConnected(architecture).to(device)
        target_net = FullyConnected(architecture).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    else:
        SIMULATION_TIME = 20
        architecture = [4,17,17,2]
        policy_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME, add_bias_as_observation=True)
        policy_net.load_state_dict((torch.load('./../CartPole-v0/DSQN-Surrogate-Gradients/initial/model.pt')))
        target_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME, add_bias_as_observation=True)
        target_net.load_state_dict(policy_net.state_dict())

    # initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    train_agent(env,policy_net,target_net,128,0.999,1.0,0.05,0.999,10,optimizer,0.001,4*10**4,device,195,100,
                num_episodes=1000,max_steps=200,gym_seed=gym_seed,torch_seed=torch_seed,random_seed=random_seed,render=False)


def train_mountain_car_dqn(spiking):
    """Test training of a DQN on MountainCar, if spiking is True a DSQN is used, else a DQN"""
    env = 'MountainCar-v0'
    torch_seed = 735
    torch.manual_seed(torch_seed)
    gym_seed = 646
    random_seed = 786
    random.seed(random_seed)
    os.chdir('Results/Test')

    # initialize policy and target net
    if not spiking:
        architecture = [2,64,64,3]
        policy_net = FullyConnected(architecture).to(device)
        target_net = FullyConnected(architecture).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    else:
        SIMULATION_TIME = 20
        architecture = [2,65,65,3]
        policy_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME,
                         add_bias_as_observation=True)
        target_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME,
                         add_bias_as_observation=True)
        target_net.load_state_dict(policy_net.state_dict())

    # initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    train_agent(env, policy_net, target_net, 128, 0.999, 1.0, 0.05, 0.999, 5,optimizer, 0.001, 10 ** 3, device, -130, 50,
                num_episodes=1000,gym_seed=gym_seed,max_steps=200,torch_seed=torch_seed,random_seed=random_seed,render=False)


def load_cartpole(spiking, save_replay):
    """Test loading a DQN on CartPole, if spiking is True a DSQN is loaded, if save_replay is True the replay dataset is saved"""
    env = 'CartPole-v0'
    os.chdir('Results/CartPole-A/SNN-Classifier')
    # initialize policy and target net
    if not spiking:
        architecture = [4, 16, 16, 2]
        policy_net = FullyConnected(architecture).to(device)
        policy_net.load_state_dict(torch.load('trained/model.pt'))

    else:
        SIMULATION_TIME = 100
        architecture = [4, 17, 17, 2]
        policy_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME,
                         add_bias_as_observation=True,encoding='constant',decoding='potential',reset='subtraction')
        policy_net.load_state_dict(torch.load('trained/model.pt'))

    load_agent(env,policy_net,device,save_replay,max_steps=500)


def train_classifier_cartpole(spiking):
    os.chdir('Results/CartPole-A/ClassifierTest')
    path_to_replay = ('./../DQN/trained/Replay_Memory')
    # initialize policy and target net
    if not spiking:
        architecture = [4, 16, 16, 2]
        policy_net = FullyConnected(architecture).to(device)

    else:
        SIMULATION_TIME = 20
        architecture = [4, 17, 17, 2]
        policy_net = SQN(architecture, device, alpha=0, beta=1, simulation_time=SIMULATION_TIME,
                         add_bias_as_observation=True)
    train_classifier(policy_net,spiking,0.0001,path_to_replay,device,iterations=2*10**5)


def load_cartpole_in_NEST():
    # specify the file, where the network is saved
    file = './Results/CartPole-A/Classifier-Converted/model.pt'

    # set hyperparameters of NEST:
    # encoding/decoding methods are limited to constant input currents and potential outputs.
    # set correct architecture
    architecture = [4, 17, 17, 2]
    # set simulation time in ms, changing the resolution is not supported in our code
    simulation_time = 100
    # this time both neuron types work similarly bad, we use pp_psc_delta
    neuron_type = 'pp_psc_delta'

    # set up network in NEST, as we load a SpyTorch trained network we have to add the bias to each observation
    nestwork = Nestwork(architecture, file, simulation_time, neuron_type=neuron_type, add_bias_as_observation=True,has_biases=False)

    # run the NEST agent and compare against the original classifier
    env = 'CartPole-v0'
    load_agent(env, nestwork, device, epsilon=0, gym_seed=1, save_replay=False,
               max_steps=500, num_episodes=100, render=True, compare_against=None)


def load_cartpole_in_pynn():
    # specify the file, where the network is saved
    file = './Results/CartPole-A/Classifier-Converted/model.pt'

    # specify the file, where the network is saved
    #file = './Results/CartPole-A/DSQN-Surrogate-Gradients/trained/model.pt'

    # set hyperparameters of NEST:
    # encoding/decoding methods are limited to constant input currents and potential outputs.
    # set correct architecture
    architecture = [4, 16, 16, 2]
    # set simulation time in ms, changing the resolution is not supported in our code
    simulation_time = 100

    # set up network in NEST, as we load a SpyTorch trained network we have to add the bias to each observation
    pynn_agent = PyNNAgent(architecture, file, simulation_time, backend='spinnaker',
                        has_biases=True, add_bias_as_observation=False)

    # run the NEST agent and compare against the original classifier
    env = 'CartPole-v0'
    load_agent(env, pynn_agent, device, epsilon=0, gym_seed=1, save_replay=False,
               max_steps=500, num_episodes=100, render=True, compare_against=None)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# call the method to test
#train_cartpole_dqn(False)
train_mountain_car_dqn(False)
#train_classifier_cartpole(True)
#load_cartpole(spiking=True,save_replay=False)
#load_cartpole_in_NEST()
#load_cartpole_in_pynn()

'''# convert a network to new format, need to copy the old DQN class to make it work
os.chdir('Results/CartPole-A/Classifier')
policy_net = FullyConnected([4,16,16,2]).to(device)
loaded_net = DQN().to(device)
loaded_net.load_state_dict(torch.load('trained/model.pt'))
policy_net.layers[0].weight.data = loaded_net.l1.weight.data
policy_net.layers[0].bias.data = loaded_net.l1.bias.data
policy_net.layers[1].weight.data = loaded_net.l2.weight.data
policy_net.layers[1].bias.data = loaded_net.l2.bias.data
policy_net.layers[2].weight.data = loaded_net.l3.weight.data
policy_net.layers[2].bias.data = loaded_net.l3.bias.data
save_model(policy_net,'copy')'''

'''# same for spiking, need to copy the old SQN class to make it
os.chdir('Results/CartPole-v0/DSQN-Surrogate-Gradients')
SIMULATION_TIME = 20
architecture = [4,17,17,2]
policy_net = DSQN(architecture,device,alpha=0,beta=1,simulation_time=SIMULATION_TIME,add_bias_as_observation=True)
loaded_net = SQN([5,17,17,2],device)
loaded_net.load(torch.load('initial/model.pt'))
policy_net.weights[0] = loaded_net.layers[0]
policy_net.weights[1] = loaded_net.layers[1]
policy_net.weights[2] = loaded_net.layers[2]
save_model(policy_net,'copy')'''

