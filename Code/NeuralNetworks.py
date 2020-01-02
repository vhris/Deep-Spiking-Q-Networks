import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """A generic fully connected network with ReLu activations, biases and no activation function for the output"""
    def __init__(self,architecture):
        """architecture needs to be a list that describes the architecture of the  network, e.g. [4,16,16,2] is a network with 4 inputs, 2 outputs and two hidden layers with 16 neurons each"""
        super(FullyConnected, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(0,len(architecture)-1):
            self.layers.append(nn.Linear(architecture[i],architecture[i+1],bias=True))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # no ReLu activation in the output layer
        x = self.layers[-1](x)
        return x

    # this function returns all intermediate results and is needed for the robust and data based normalization methods for conversion to a spiking network
    def forward_return_all(self, x):
        all_neurons_output = []
        for i in range(0,len(self.layers)-1):
            x = F.relu(self.layers[i](x))
            all_neurons_output.append(x)
        # no ReLu activation in the output layer
        x = self.layers[-1](x)
        all_neurons_output.append(x)
        return all_neurons_output

