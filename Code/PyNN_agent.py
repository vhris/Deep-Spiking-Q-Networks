# Load any spiking network in PyNN and simulate it using spiNNaker or NEST as backend

from scipy.special import erf
import pylab
import pyNN.nest as nest_sim
import spynnaker8
from spynnaker8 import extra_models
import spynnaker8 as spin_sim
import torch
import numpy as np

class PyNNAgent:

    def __init__(self, architecture, file, simulation_time, backend='spinnaker',add_bias_as_observation=False):
        """

        :param architecture: shape of the network that is loaded, should be a list of the form [size_input, size_hidden_1,...,size_hidden_n, size_output]
        :param file: the file, where the network which is loaded is saved
        :param simulation_time: default simulation time for the agent, can be changed for individual simulations
        :param backend: spinnaker or nest or pynn (for native pynn models)
        :param add_bias_as_observation: this option is for loading a network trained using SpyTorch as SpyTorch allows no hidden layer biases. If true a 1 is added as constant input
        :param has_biases: whether the loaded network has biases, if yes loading looks slightly different
        """
        self.architecture = architecture
        self.file = file
        self.simulation_time = simulation_time
        self.backend = backend
        self.add_bias_as_observation = add_bias_as_observation
        # load weights and biases from file
        layers = torch.load(self.file)
        weights = layers[0]
        biases = layers[1]
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

        # setup the network
        self.setup()


    def setup(self):

        # set up layers according to backend
        self.layers = []
        if self.backend == 'spinnaker':
            spin_sim.setup(timestep=1.0, time_scale_factor=1.0, min_delay=1.0, max_delay=None)
            # set neuron hyperparameters
            # tau_m should not be set too high, because it can easily lead to an overflow
            ctx_parameters = {'cm': 1.0, 'i_offset': 0.0, 'tau_m': 100.0, 'tau_refrac': 0.0, 'v_reset': 0.0,
                              'v_rest': 0.0, 'v_thresh': 1.0}
            tc_parameters = ctx_parameters.copy()
            neuron_type = extra_models.IFCurDelta(**tc_parameters)
            for l in range(0,len(self.weights)):
                self.layers.append(spin_sim.Population(self.architecture[l+1], neuron_type, initial_values={'v': 0.0}))
            # set output layer threshold: must be set high enough s.t. output neurons do not spike, must not be set too high to avoid overflow
            self.layers[-1].set(v_thresh=10000.0)
        elif self.backend == 'nest':
            nest_sim.setup(timestep=1.0,  min_delay=1.0)
            # set neuron type, currently only supports iaf_psc_delta and one set of hyperparameters
            neuron_type = nest_sim.native_cell_type('iaf_psc_delta')
            for l in range(0, len(self.weights)-1):
                self.layers.append(nest_sim.Population(self.architecture[l+1],neuron_type(V_th=1.0, t_ref=0.0, V_min=-np.inf, C_m=1.0, V_reset=0.0,
                                                         tau_m=10.0 ** 10, E_L=0.0, I_e=0.0)))
                self.layers[l].initialize(V_m=0.0)
            # set up output layer seperately with infinite threshold
            self.layers.append(nest_sim.Population(self.architecture[-1],neuron_type(V_th=np.inf, t_ref=0.0, V_min=-np.inf, C_m=1.0, V_reset=0.0,
                                                     tau_m=10.0 ** 10, E_L=0.0, I_e=0.0)))
            self.layers[-1].initialize(V_m=0.0)
        elif self.backend == 'pynn':
            raise Exception('Current implementation does not support native pynn types. '
                            'Code after this exception acts as an example on how native pynn types are used and can be modified.'
                            'Additionally, the backend still needs to be specified and is set to nest in the code below.')
            # the following outcommented lines show how a native pynn type would be used
            ''' 
            nest_sim.setup(timestep=1.0, time_scale_factor=1.0, min_delay=1.0, max_delay=None)
            # conductive exponential neuron as an example
            ctx_parameters = {'cm': 1.0, 'e_rev_E': 0.0, 'e_rev_I': -65.0, 'i_offset': 0.0, 'tau_m': 10000.0, 'tau_refrac': 0.0, 'tau_syn_E': 0.01, 'tau_syn_I': 0.01, 'v_reset': -65.0, 'v_rest': -65.0, 'v_thresh': -64.0}
            #tc_parameters = ctx_parameters.copy()
            neuron_type = sim.IF_cond_exp(**tc_parameters)
            for l in range(0,len(self.weights)):
                self.layers.append(sim.Population(self.architecture[l+1], neuron_type, initial_values={'v': 0.0}))
            # output layer should not spike
             self.layers[-1].set(v_thresh=np.inf)
            '''
        else:
            raise Exception('backend not supported')

        # connect layers (input layer is connected to first in simulate by setting offsets
        for l in range(0, len(self.layers)-1):
            excitatory_connections = []
            inhibitory_connections = []
            # 1 and 2
            for i in range(0, len(self.layers[l])):
                for j in range(0, len(self.layers[l+1])):
                    if self.weights[l+1][i][j].detach().item() >= 0:
                        excitatory_connections.append((i, j, self.weights[l+1][i][j].detach().item(),1.0))
                    else:
                        inhibitory_connections.append((i, j, self.weights[l+1][i][j].detach().item(),1.0))
            # when using NEST or native PyNN models, connect has to be called explicitly, when using sPiNNaker connecting happens automatically
            if self.backend == 'pynn' or self.backend == 'nest':
                excitatory_connector = nest_sim.FromListConnector(excitatory_connections, column_names=["weight", "delay"])
                excitatory_projection = nest_sim.Projection(self.layers[l], self.layers[l + 1],connector=excitatory_connector)
                excitatory_connector.connect(excitatory_projection)
            elif self.backend == 'spinnaker':
                excitatory_connector = spin_sim.FromListConnector(excitatory_connections, column_names=["weight", "delay"])
                excitatory_projection = spin_sim.Projection(self.layers[l], self.layers[l + 1],
                                                       connector=excitatory_connector)

            # when using NEST or native PyNN models, connect has to be called explicitly, when using sPiNNaker connecting happens automatically
            if self.backend == 'pynn' or self.backend == 'nest':
                inhibitory_connector = nest_sim.FromListConnector(inhibitory_connections, column_names=["weight", "delay"])
                inhibitory_projection = nest_sim.Projection(self.layers[l], self.layers[l + 1], inhibitory_connector,
                                                       receptor_type='inhibitory')
                inhibitory_connector.connect(inhibitory_projection)
            elif self.backend == 'spinnaker':
                inhibitory_connector = spin_sim.FromListConnector(inhibitory_connections, column_names=["weight", "delay"])
                inhibitory_projection = spin_sim.Projection(self.layers[l], self.layers[l + 1], inhibitory_connector,
                                                       receptor_type='inhibitory')

        # set biases (constant input currents)
        for l in range(0,len(self.layers)):
            for i in range(0,len(self.layers[l])):
                #use i_offset on spynnaker instead of DCSource
                offset = self.biases[l][i].detach().item()
                if self.backend == 'nest':
                    self.layers[l][i:i+1].set(I_e=offset)
                elif self.backend == 'pynn' or self.backend=='spinnaker':
                    self.layers[l][i:i+1].set(i_offset=offset)

        # record the potentials of the last layer
        if self.backend == 'nest' or self.backend == 'pynn':
            self.layers[-1].record('V_m')
        elif self.backend == 'spinnaker':
            self.layers[-1].record('v')


    def forward(self,input, timesteps=None):
        # if timesteps not specified, set to default simulation time
        if timesteps is None:
            timesteps = self.simulation_time
        # first reset the network
        if self.backend == 'nest' or self.backend == 'pynn':
            nest_sim.reset()
        elif self.backend == 'spinnaker':
            spin_sim.reset()
        # set input generators to correct current
        if self.add_bias_as_observation:
            bias = torch.ones(1, dtype=torch.float)
            input = torch.cat((input, bias), dim=0)
        # set offsets to correct current
        # save the original offsets/biases
        bias = []
        for i in range(0, len(self.layers[0])):
            if self.backend == 'pynn' or self.backend == 'spinnaker':
                offset = self.layers[0][i:i+1].get('i_offset')[0]
                bias.append(offset)
                # add the inputs multiplied by their respective weights to the constant input of the first hidden layer
                for j in range(0,len(input)):
                    offset += input[j].detach().item()*self.weights[0][j][i].detach().item()
                self.layers[0][i:i+1].set(i_offset=offset)
            elif self.backend == 'nest':
                offset = self.layers[0][i:i+1].get('I_e')
                bias.append(offset)
                # add the inputs multiplied by their respective weights to the constant input of the first hidden layer
                for j in range(0, len(input)):
                    offset += input[j].detach().item() * self.weights[0][j][i].detach().item()
                self.layers[0][i:i + 1].set(I_e=offset)

        # simulate
        if self.backend == 'nest' or self.backend == 'pynn':
            nest_sim.run(timesteps)
        elif self.backend == 'spinnaker':
            spin_sim.run(timesteps)

        potentials = self.layers[-1].get_data().segments[-1].analogsignals[0][-1]
        # restore original bias in the offset
        for i in range(0, len(self.layers[0])):
            if self.backend == 'pynn' or self.backend == 'spinnaker':
                self.layers[0][i:i + 1].set(i_offset=bias[i])
            elif self.backend == 'nest':
                self.layers[0][i:i + 1].set(I_e=bias[i])
        #  return torch tensor to be compatible with other agents
        return torch.tensor(potentials,dtype=torch.float)