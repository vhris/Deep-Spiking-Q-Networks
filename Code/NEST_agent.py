# Load any spiking network in NEST

import torch

# this imports should come before importing nest
#from sklearn.svm import LinearSVC
from scipy.special import erf
import pylab
# importing nest works only if pycharm community is started from the command line (command: pycharm-community), because else .bashrc environment variables are not set
import nest
import numpy as np


class Nestwork:

    def __init__(self,architecture,file, simulation_time,neuron_type='iaf_psc_delta', add_bias_as_observation=False, has_biases=True):
        """

        :param architecture: shape of the network that is loaded, should be a list of the form [size_input, size_hidden_1,...,size_hidden_n, size_output]
        :param file: the file, where the network which is loaded is saved
        :param simulation_time: default simulation time for the agent, can be changed for individual simulations
        :param neuron_type: type of the neuron, we tested iaf_psc_delta (LIF neuron) and pp_psc_delta (LIF neuron with adaptive thresholds and stochastic firing)
        :param add_bias_as_observation: this option is for loading a network trained using SpyTorch as SpyTorch allows no hidden layer biases. If true a 1 is added as constant input
        :param has_biases: whether the loaded network has biases, if yes loading looks slightly different
        """
        self.architecture = architecture
        self.add_bias_as_observation = add_bias_as_observation
        self.input_size = architecture[0] + self.add_bias_as_observation
        self.file = file
        self.simulation_time = simulation_time
        # load weights and biases from file
        layers = torch.load(self.file)
        # if network has biases, split network into weights and biases
        if has_biases:
            weights = layers[0]
            biases = layers[1]
        # else network has only weights
        else:
            weights = layers
            biases = [None]*len(architecture)
        self.weights = []
        self.biases = []
        for l in range(0, len(weights)):
            self.weights.append([])
            self.biases.append([])
            self.weights[l] = weights[l].detach().clone()
            if biases[l] is not None:
                self.biases[l] = biases[l].detach().clone()
            else:
                self.biases[l] = biases[l]

        # setup the network in NEST
        self.setup(neuron_type)

    def setup(self, neuron_type):
        # set up neurons
        # create input layer
        self.inputs = nest.Create("dc_generator",n=self.input_size)

        # set up the hidden layers depending on the neuron type
        self.layers = []
        if neuron_type == 'iaf_psc_delta':
            # set up neuron hyperparameters
            hyper_iaf_delta = {"V_th": 1.0, "t_ref": 0.0, "V_min": -np.inf, "C_m": 1.0, "V_reset": 0.0,
                               "tau_m": 10.0 ** 10,
                               "E_L": 0.0, "V_m": 0.0, "I_e": 0.0}
            # create hidden layers
            for l in range(0,len(self.weights)-1):
                # create first layer
                self.layers.append(nest.Create(neuron_type, n=self.architecture[l+1]))
                nest.SetStatus(self.layers[l], hyper_iaf_delta)

        elif neuron_type == 'pp_psc_delta':
            # set up neuron hyperparameters
            hyper_pp_delta = {"V_m": 0.0, "C_m": 1.0, "tau_m": 10.0 ** 10, "q_sfa": 1.0, "tau_sfa": 10.0 ** 10,
                              "dead_time": 1e-8,
                              "dead_time_random": False, "dead_time_shape": 1, "t_ref_remaining": 0.0, "I_e": 0.0,
                              "c_1": 10.0 ** 10, "c_2": 10.0, "c_3": 5.0, "with_reset": False}
            # create hidden layers
            for l in range(0, len(self.weights)-1):
                self.layers.append(nest.Create("pp_psc_delta", n=self.architecture[l+1]))
                nest.SetStatus(self.layers[l],hyper_pp_delta)

        # create output layer independent of the neuron type
        hyper_output_layer = {"V_th": np.inf, "t_ref": 0.0, "V_min": -np.inf, "C_m": 1.0, "V_reset": 0.0,
                              "tau_m": 10.0 ** 10,
                              "E_L": 0.0, "V_m": 0.0, "I_e": 0.0}
        self.layers.append(nest.Create("iaf_psc_delta", n=self.architecture[-1]))
        # the last layer never spikes, only the membrane potentials are compared
        nest.SetStatus(self.layers[-1],hyper_output_layer)

        # set up biases as constant input currents
        for l in range(0,len(self.biases)):
            if self.biases[l] is not None:
                for i in range(0, len(self.biases[l])):
                    nest.SetStatus([self.layers[l][i]], {"I_e": self.biases[l][i].detach().item()})

        # connect layers
        # input and 1
        for i in range(0, len(self.inputs)):
            for j in range(0, len(self.layers[0])):
                syn_dict_ex = {"weight": self.weights[0][i][j].detach().item()}
                nest.Connect([self.inputs[i]], [self.layers[0][j]], syn_spec=syn_dict_ex)
        for l in range(0,len(self.layers)-1):
            # connect l-th with l+1-th layer
            for i in range(0, len(self.layers[l])):
                for j in range(0,len(self.layers[l+1])):
                    syn_dict_ex = {"weight": self.weights[l+1][i][j].detach().item()}
                    nest.Connect([self.layers[l][i]], [self.layers[l+1][j]], syn_spec=syn_dict_ex)

        # connect multimeter (records membrane potential) to the output layer, one multimeter for each output
        self.multimeters = []
        for i in range(0,self.architecture[-1]):
            self.multimeters.append(nest.Create("multimeter"))
            nest.SetStatus(self.multimeters[i], {"withtime": True, "record_from": ["V_m"]})
            nest.Connect(self.multimeters[i], [self.layers[-1][i]])

    def forward(self,input, timesteps=None):
        # if timesteps not specified, set to default simulation time
        if timesteps is None:
            timesteps = self.simulation_time
        # first reset the network
        nest.ResetNetwork()
        # set input generators to correct current
        if self.add_bias_as_observation:
            bias = torch.ones(1,dtype=torch.float)
            input = torch.cat((input,bias),dim=0)
        for i in range(0,len(self.inputs)):
            nest.SetStatus([self.inputs[i]], {"amplitude": input[i].detach().item()})

        # simulate
        nest.Simulate(timesteps)
        #get recorded value from multimeter
        outputs = []
        for i in range(0,len(self.multimeters)):
            dmm = nest.GetStatus(self.multimeters[i])[0]
            Vms = dmm["events"]["V_m"]
            outputs.append(Vms[-1])

        # return a tensor of the outputs to be compatible with the other agents
        return torch.tensor(outputs,dtype=torch.float)

