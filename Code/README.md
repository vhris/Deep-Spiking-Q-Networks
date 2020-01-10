# Code structure
This file contains the main code of the project and is structured in the following way:
  - \_\_init\_\_.py Allows importing this folder as a module.
  - load_agent.py Loads a pre-trained agent described by a neural network and simulates it on a gym environment. Additionally,
                  a second agent can be specified against which the first is compared.
                  Optionally saves the replay memory of the simulated agent.
  - NEST_agent.py Creates a NEST agent based on a previously trained or converted neural network.
  - NeuralNetworks.py Implements the neural networks using PyTorch.
  - PyNN_agent.py Creates an agent in PyNN based on a previously trained or converted neural network. Uses NEST or SpiNNaker as backend
  - replay_memory.py Implements the classes replay_memory and replay_dataset used for training a DQN and for converting a DQN respectively
  - SQN.py Implements our spiking neural network class which is based on the implementation of SpyTorch [1].
            Class is not limited to Q-networks, but can implement any other type of spiking neural network.
            So far is limited to fully connected networks.
            Contains method weight_conversion which converts the weights of a neural network in order to be used in a SNN. 
            Conversion uses the method from Rueckauer et al. [2]
  - train_agent.py Trains a DQN or DSQN on a gym environment.
  - train_classifier.py Trains a spiking or non-spiking classifier on the replay memory.
  - utils. Contains various functions for plotting and saving.

How to use the code is explained in detail in the CartPole experiment notebooks.

[1] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate Gradient Learning in Spiking
	Neural Networks. IEEE SPM WHITE PAPER FOR THE SPECIAL ISSUE ON NEUROMORPHIC COMPUTING, 28 January 2019.

[2] Bodo Rueckauer, Iulia-Alexandra Lungu, Yuhuang Hu, and Michael Pfeiffer. Theory and Tools for the Conversion of Analog to Spiking Convolutional Neural Networks. December 2016.
