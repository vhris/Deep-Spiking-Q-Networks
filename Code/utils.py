import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from IPython.display import clear_output
import matplotlib
from matplotlib import pyplot as plt
import collections
# set up matplotlib for iPython (Jupyter Notebook)
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def plot_durations(episode_durations, plot_mean=True,figure_number=1,title='Training...',xlabel='Episode',ylabel='Reward',default_ones=False,max_steps=0):
    """Plots the duration length for the episodes of a gym environment"""
    plt.figure(figure_number)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(durations_t.numpy())

    if plot_mean:
        # Take 100 (or len(epsisodes)) episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            x_values = np.linspace(100, len(durations_t), len(durations_t)-99)
            plt.plot(x_values,means.numpy())

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    else:
        plt.pause(0.000001)  # pause a bit so that plots are updated


def plot_mean(episode_durations):
    """plots the average over all episodes"""
    # Plot average
    durations = np.array(episode_durations)
    mean = np.mean(durations)
    plt.plot((0, len(episode_durations)-1), (mean, mean), 'orange')
    plt.title('Mean: ' + str(mean))


def save_model(model, directory):
    """Saves a pytorch model"""
    cwd = os.getcwd()
    os.makedirs(directory)
    os.chdir(directory)
    torch.save(model.state_dict(), 'model.pt')
    os.chdir(cwd)


def save_hyperparameters(discount_factor, learning_rate, epsilon_start, epsilon_end, epsilon_decay, replay_memory_size,
                         sample_size, target_update_frequency, others=''):
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
        file.write(others)


def plot_accuracies(means,figure_number = 2,xlabel='Iterations',ylabel='Accuracy',title='Accuracies'):
    """Plots the duration length for the episodes of a gym environment"""
    plt.figure(figure_number)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(np.array(means),'orange')
    plt.ylim((-0.1,1.1))
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    else:
        plt.pause(0.001)  # pause a bit so that plots are updated