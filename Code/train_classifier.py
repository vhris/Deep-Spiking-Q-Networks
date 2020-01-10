# Test

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .replay_memory import replay_dataset
from .train_agent import save_model

# set up matplotlib for iPython (Jupyter Notebook)
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot(means,fig,figure_number = 1,xlabel='',ylabel='',title=''):
    """Plots the duration length for the episodes of a gym environment"""
    fig.add_subplot(1,2,figure_number)
    plt.title(title)
    plt.xlabel(xlabel)
    # only plot the y label if ipython is True, else the plot looks to cramped
    if is_ipython:
        plt.ylabel(ylabel)
    plt.plot(np.array(means),'orange')
    plt.ylim((0,1))
    if not is_ipython:
        plt.pause(0.000001)  # pause a bit so that plots are updated


def train_classifier(neural_net,spiking,learning_rate,path_to_replay,device,batch_size=50,iterations=2*10**5):
    # Save initial weights
    save_model(neural_net, 'initial')
    # train classifier on the q values
    optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)
    policy = replay_dataset(path_to_replay,device)

    losses = []
    mean_losses = []
    accuracies = []
    mean_accuracies = []

    for i in range(iterations):
        batch = policy.next_batch(batch_size)
        # train
        # Compute loss
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()
        predictions = neural_net.forward(batch[0])
        log_softmax = log_softmax_fn(predictions)
        loss = loss_fn(log_softmax, batch[1])

        mismatches = 0
        for j in range(0, len(batch[1])):
            if torch.argmax(predictions[j]) != batch[1][j]:
                mismatches += 1
        accuracies.append(1 - mismatches / len(batch[1]))

        losses.append(loss.detach().clone())
        if len(losses) > 1000:
            losses.pop(0)
            accuracies.pop(0)
        # calculate mean only every 100 episodes
        if i % 100 == 0:
            mean_losses.append(sum(losses) / len(losses))
            mean_accuracies.append(sum(accuracies) / len(accuracies))
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # plot every 1000 iterations
        if i % 1000 == 0:
            # clear output before plotting new figures
            if is_ipython:
                display.clear_output(wait=True)
            print('current iteration: ', i)
            if i>1:
                plt.close("all")
            fig = plt.figure(0)
            plt.tight_layout()
            plot(mean_losses,fig, figure_number=1, title='Loss', xlabel='Iteration in hundreds', ylabel='Loss')
            plot(mean_accuracies,fig,figure_number=2, title='Accuracy', xlabel='Iteration in hundreds', ylabel='Accuracy')
            plt.tight_layout()
            if is_ipython:
                display.display(plt.gcf())


    # Save learned weights
    save_model(neural_net, 'trained')
    fig.savefig('AccuracyAndLoss.png')
    plt.ioff()
    # report loss and accuracy over the last 1000 episodes
    print('Loss: ',sum(losses[-1000::]) / len(losses[-1000::]))
    print('Accuracy: ',sum(accuracies[-1000::]) / len(accuracies[-1000::]))