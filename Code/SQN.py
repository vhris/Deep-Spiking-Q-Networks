# this file contains inplementations of a SpyTorch SQN and training methods for it

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .replay_memory import replay_dataset


class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class SQN(nn.Module):

    def __init__(self,network_shape,device,alpha,beta, weight_scale=1,encoding='constant',decoding='potential',threshold=1,simulation_time=100,reset='subtraction',
                 two_input_neurons=False, add_bias_as_observation=False, has_biases=True):
        #TODO add output method 'time_to_first_spike'?
        #TODO add two_output_neurons
        """Args:
            network_shape: shape of the network given in the form [5,17,17,2] for 5 input neurons, two hidden layers with 17 neurons and 2 output neurons
            device: device for the torch tensors
            alpha: synapse decay
            beta: membrane decay
            weight_scale: determines the standard deviaiton of the random weights
            encoding: 'constant','equidistant' or 'poisson' for the three possible input methods
            decoding: 'potential' or 'spikes' for the two different output methods, also 'first_spike' is possible which returns once the fiirst output neuron spikes
            threshold: 'threshold' when a spike occurs
            simulation_time: number of time steps to be simulated
            reset: either 'subtraction' or 'zero'
            two_input_neurons: for input methods 'equidistant' and 'poisson' if one output neuron is to be used for negative and positive inputs each
            add_bias_as_observation: this option is for training using SpyTorch as SpyTorch allows no hidden layer biases. If true a 1 is added as constant input
            has_biases: whether the network has biases, if False this is a special case for load"""
        self.alpha = alpha
        self.beta = beta
        self.encoding = encoding
        self.decoding = decoding
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.device = device
        self.input_size = network_shape[0]
        self.reset = reset
        self.add_bias_as_observation = add_bias_as_observation
        self.two_input_neurons = two_input_neurons
        self.has_biases= has_biases

        if self.add_bias_as_observation:
            # add one more neuron to the architecture at the input, because the bias acts as an additional input
            network_shape[0] += 1

        self.weights = []
        self.bias = []
        for i in range(0,len(network_shape)-1):
            self.weights.append(torch.empty((network_shape[i], network_shape[i+1]), device=device, dtype=torch.float, requires_grad=True))
            torch.nn.init.normal_(self.weights[i], mean=0.0, std=weight_scale / np.sqrt(network_shape[i]))
            # initialize all biases with None, SpyTorch does not support biases.
            self.bias.append(None)

        self.spike_fn = SuperSpike.apply

        if self.two_input_neurons:
            self.weights[0] = torch.cat([self.weights[0],torch.neg(self.weights[0])])
            self.input_size *= 2

    def forward(self, input_data):
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

        # if two input neurons is True, double the input size, where the first set of values represents the positive inputs and the second set the negative inputs.
        if self.two_input_neurons:
            input_data = torch.cat([input_data * (input_data > 0), torch.neg(input_data * (input_data < 0))])
        # reshape input such that it is in the form (batch_size, input_dimenstion) and not (input_dimension,) or (input_dimension)
        if input_data.shape == (self.input_size,) or input_data.dim() == 1:
            input_data = input_data.reshape(1, self.input_size)
        batch_size = input_data.shape[0]

        if self.add_bias_as_observation:
            bias = torch.ones((batch_size,1),device=self.device,dtype=torch.float)
            input_data = torch.cat((input_data,bias),dim=1)

        # reset the array for membrane potential and synaptic variables
        syn = []
        mem = []
        for l in range(0, len(self.weights)):
            syn.append(torch.zeros((batch_size, self.weights[l].shape[1]), device=self.device, dtype=torch.float))
            mem.append(torch.zeros((batch_size, self.weights[l].shape[1]), device=self.device, dtype=torch.float))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []
        # Additionally we define a list that counts the spikes in the output layer, if the output method 'spikes' is used
        if self.decoding == 'spikes':
            spk_count = torch.zeros((batch_size,self.weights[-1].shape[1]),device=self.device,dtype=torch.long)

        if self.encoding == 'equidistant':
            # spike counter is used to count the number of spike for each input neuron so far
            spike_counter = torch.ones_like(input_data)
            fixed_distance = 1 / input_data

        # Here we loop over time
        for t in range(self.simulation_time):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            if self.encoding == 'constant':
                input = input_data.detach().clone()
            elif self.encoding == 'poisson':
                #generate poisson distributed input
                spike_snapshot = torch.tensor(np.random.uniform(low=0, high=1, size=input_data.shape),device=self.device)
                input = (spike_snapshot <= input_data).float()
            elif self.encoding == 'equidistant':
                # generate fixed number of equidistant spikes
                input = (torch.ones_like(input_data)*t == torch.round(fixed_distance*spike_counter)).float()
                spike_counter += input
            else:
                raise Exception('Encoding Method '+str(self.encoding)+' not implemented')

            # loop over layers
            for l in range(len(self.weights)):
                # define impulse
                if l == 0:
                    h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                else:
                    h = torch.einsum("ab,bc->ac", [spk_rec[len(spk_rec) - 1][l - 1], self.weights[l]])
                # add bias
                if self.bias[l] is not None:
                    h += self.bias[l]

                # calculate the spikes for all layers (decoding='spikes' or 'first_spike') or for all but the last layer (decoding='potential')
                if self.decoding == 'spikes' or self.decoding == 'first_spike' or  l<len(self.weights)-1:
                    mthr = mem[l] - self.threshold
                    out = self.spike_fn(mthr)
                    rst = torch.zeros_like(mem[l],device=self.device)
                    c = (mthr > 0)
                    rst[c] = torch.ones_like(mem[l],device=self.device)[c]
                    # count the spikes in the output layer
                    if self.decoding == 'spikes' and l == len(self.weights)-1:
                        spk_count = torch.add(spk_count,out)
                else:
                    # else reset is 0 (= no reset)
                    c = torch.zeros_like(mem[l],dtype=torch.bool,device=self.device)
                    rst = torch.zeros_like(mem[l],device=self.device)

                # calculate the new synapse potential
                new_syn = self.alpha * syn[l] + h
                # calculate new membrane potential
                if self.reset == 'subtraction':
                    new_mem = self.beta * mem[l] + syn[l] - rst
                elif self.reset == 'zero':
                    new_mem = self.beta * mem[l] + syn[l]
                    new_mem[c] = 0

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[len(mem_rec) - 1].append(mem[l])
                spk_rec[len(spk_rec) - 1].append(out)

                if self.decoding == 'first_spike' and l==len(self.weights)-1:
                    if torch.sum(out)>0:
                        return out
        if self.decoding == 'potential':
            # return the final recorded membrane potential (len(mem_rec)-1) in the output layer (-1)
            return mem_rec[len(mem_rec)-1][-1]
        if self.decoding == 'spikes':
            # return the sum over the spikes in the output layer
            return spk_count
        else:
            raise Exception('Decoding Method '+str(self.decoding)+' not implemented')


    def load_state_dict(self,layers):
        """Method to load weights and biases into the network"""
        # if network has biases, split network into weights and biases
        if self.has_biases:
            weights = layers[0]
            biases = layers[1]
        # else network has only weights
        else:
            weights = layers
            biases = [None] * len(self.bias)
        for l in range(0,len(weights)):
            self.weights[l] = weights[l].detach().clone().requires_grad_(True)
            if biases[l] is not None:
                self.bias[l] = biases[l].detach().clone().requires_grad_(True)
            else:
                self.bias[l] = biases[l]

        # if two input neurons are used, split the weights into two groups, one for positive and one for negative inputs.
        # the weights for the negative inputs are multiplied by minus 1.
        if self.two_input_neurons:
            self.weights[0] = torch.cat([self.weights[0],torch.neg(self.weights[0])])

    def load(self,weights,biases):
        """Analog to previous method, but weights and biases are passed directly
        it is used for converting networks.
        For some reason, when converting, weights are saved in transposed form, so we need to trandpose them back to the correct shape."""
        for l in range(0,len(weights)):
            self.weights[l] = weights[l].detach().clone().transpose(0,1).requires_grad_(True)
            if biases[l] is not None:
                self.bias[l] = biases[l].detach().clone().requires_grad_(True)
            else:
                self.bias[l] = biases[l]

        # if two input neurons are used, split the weights into two groups, one for positive and one for negative inputs.
        # the weights for the negative inputs are multiplied by minus 1.
        if self.two_input_neurons:
            self.weights[0] = torch.cat([self.weights[0],torch.neg(self.weights[0])])

    def state_dict(self):
        """Method to copy the layers of the SQN. Makes explicit copies, no references."""
        weights_copy = []
        bias_copy = []
        for l in range(0, len(self.weights)):
            weights_copy.append(self.weights[l].detach().clone())
            if self.bias[l] is not None:
                bias_copy.append(self.bias[l].detach().clone())
            else:
                bias_copy.append(self.bias[l])
        return weights_copy, bias_copy

    def parameters(self):
        parameters = []
        for l in range(0,len(self.weights)):
            parameters.append(self.weights[l])
            if self.bias[l] is not None:
                parameters.append(self.bias[l])
        return parameters


def weight_conversion(model,weights,bias,device, normalization_method = 'robust',ppercentile=0.99,path_to_replay=None):
    """Args:
            model: the model that is converted
            weights: a list, where the first element is the weights of the first layer of the model and so on
            bias: as weights, but for the bias
            device: device which torch is using
            normalization_method: 'robust' or 'model' or 'data' for the method from Rueckauer and the two from Diehl respectively
            pp_percentile needs to be set to the desired percentile, for 'data' it is automatically set to 1
        Returns: converted weights and parameters"""

    if normalization_method == 'data':
        ppercentile = 1.0

    # Get weights from trained network
    converted_weights = weights
    converted_bias = bias

    if normalization_method == 'model':
        # model based normalization
        prev_factor = 1
        for l in range(len(converted_weights)):
            max_pos_input = 0
            # Find maximum input for this layer
            for o in range(converted_weights[l].shape[0]):
                input_sum = 0
                for i in range(converted_weights[l].shape[1]):
                    input_sum += max(0, converted_weights[l][o,i])
                if converted_bias[l] is not None:
                    input_sum += max(0, converted_bias[l][o])
                max_pos_input = max(max_pos_input, input_sum)
            # get the maximum weight in the layer, in case all weights are negative, max_pos_input would be zero, so we use the max weight to rescale instead
            max_wt = torch.max(converted_weights[l])
            if converted_bias[l] is not None:
                max_bias = torch.max(converted_bias[l])
                max_wt = torch.max(max_wt, max_bias)
            scale_factor = max(max_wt, max_pos_input)
            # Rescale all weights
            applied_factor = scale_factor/prev_factor
            converted_weights[l] = converted_weights[l] / applied_factor
            if converted_bias[l] is not None:
                converted_bias[l] = converted_bias[l] / scale_factor
            prev_factor = scale_factor

    if normalization_method == 'robust' or normalization_method == 'data':
        # use training set to find max_act for each neuron
        if path_to_replay is None:
            raise Exception("Data based training method chosen, but no path to replay dataset provided")
        training_set = replay_dataset(path_to_replay,device)
        activations = []
        for l in range(0, len(converted_weights)):
            activations.append([])
        max_it = 20000
        current_it = 0
        for obs in training_set.observations:
            current_it += 1
            if current_it > max_it:
                break
            output = model.forward_return_all(obs)
            for x in range(0, len(activations)):
                activations[x].append(torch.max(output[x]))

        previous_factor = 1
        for l in range(len(converted_weights)):
            # get the p-percentile of the activation
            pos_inputs = activations[l]
            pos_inputs.sort()
            max_act = pos_inputs[int(ppercentile * (len(pos_inputs) - 1))]
            # get the maximum weight in the layer
            max_wt = torch.max(converted_weights[l])
            if converted_bias[l] is not None:
                max_bias = torch.max(converted_bias[l])
                max_wt = torch.max(max_wt, max_bias)
            scale_factor = max(max_wt, max_act)

            applied_factor = scale_factor / previous_factor
            # rescale weights
            converted_weights[l] = converted_weights[l] / applied_factor

            # rescale bias
            if converted_bias[l] is not None:
                converted_bias[l] = converted_bias[l] / scale_factor
            previous_factor = scale_factor

    else:
        raise Exception('Unsupported normalization method '+str(normalization_method))

    return converted_weights,converted_bias
