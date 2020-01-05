# Deep-Spiking-Q-Networks
This repository contains the code for the Master's thesis "Deep Spiking Q Networks" conducted at the Robotics chair of the Technical University of Munich submitted in January 2020.
It implements training methods of spiking neural networks applied to three OpenAIGym environments (CartPole, MountainCar, and Breakout). The methods implemented are three different conversion techniques:
  1. Directly converting a DQN using the conversion methods from Rueckauer et al. [1] 
  2. Training a classifier to learn the policy of the DQN and then converting the classifier using the same method.
  3. Training a spiking classification network on the policy of the DQN using Backpropagation with surrogate gradients.
Further, it implements direct training of a spiking DQN (DSQN) using Backpropagation with surrogate gradients as well. We use the SpyTorch [2] framework to implement surrogate gradients.
Additionally, we show how to run load our experiments in the neural network simulators NEST and PyNN. For PyNN we implement both NEST and SpiNNaker neuromorphic hardware as backend.
Our thesis results are presented in a series of jupyter notebooks. To learn how to use our code, the CartPole experiments can be used as tutorials.
## Repository Structure
The repository is structured in the following way
  1. Code: Code of the repository.
  2. Experiments as Jupyter Notebooks, CartPole experiments can be used as a tutorial on how to use the code.
  4. Presentations: Mid- and Endterm presentations held at TUM.
  5. Results: Saved networks used in the thesis.
  6. Deep-Spiking-Q-Networks.pdf: Thesis
  7. requirements.txt: Library versions of the virtual environment we used (created with "pip3 freeze"). Additionally we use
     NEST 2.16.0 which need to be downloaded from https://www.nest-simulator.org/download/. The python version we used is
     3.5.
The folders code, experiments, and results contain their own Readme that explains their structure in more detail.

## Prerequisites
To run the code it should be sufficient to install gym, torch, matplotlib, NEST, PyNN, and SpiNNaker which should install all required dependencies as well. The parts of the code not relying on NEST, PyNN and SpiNNaker can be run without installing the respective libraries if the corresponding imports are deleted, else these imports cause errors. For the specific versions we used see requirements.txt.

## Tutorials
The CartPole experiments are written as tutorials which demonstrate how to use our code.

## Acknowledgements
Thanks to my advisor Mahmoud Akl and to my supervisor Prof. Dr.-Ing. Alois Christian Knoll for making this thesis possible. Further, credit goes to the creators of SpyTorch [2] and Claus Meschede [3] on whose code this repository is based.

## Sources
[1] Bodo Rueckauer, Iulia-Alexandra Lungu, Yuhuang Hu, and Michael Pfeiffer. Theory and Tools for the Conversion of Analog to Spiking Convolutional Neural Networks. December 2016.
[2] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate Gradient Learning in Spiking
	Neural Networks. IEEE SPM WHITE PAPER FOR THE SPECIAL ISSUE ON NEUROMORPHIC COMPUTING, 28 January 2019.
[3] Claus Meschede. Training Neural Networks for Event-Based
	End-to-End Robot Control. Master thesis at the Technical University of Munich, July 2017.
These and all other sources used are listed in the bibliography of the thesis.

