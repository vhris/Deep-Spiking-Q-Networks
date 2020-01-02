import gym
import numpy as np
import torch
import warnings
import statistics
from itertools import count
import matplotlib.pyplot as plt
import random

from .replay_memory import ReplayMemory
from .utils import plot_durations,plot_mean

#TODO unify this with train_agent?

def load_agent(environment,policy_net, device,epsilon=0,gym_seed=None,save_replay=False,replay_size=2*10**4,max_steps=None,num_episodes=100,render =True, compare_against=None):
    # set up environment
    env = gym.make(environment)
    if gym_seed is not None:
        env.seed(gym_seed)
    if max_steps is not None:
        env._max_episode_steps = max_steps

    # keep track of rewards
    episode_rewards = []

    # initialize replay memory
    if save_replay:
        memory = ReplayMemory(replay_size)
        memory_saved = False

    # if compare against is not None we compare each prediction of the agent with the compare against agent
    # we count the number of "correct" classifications, where both agents predict the same and also count the number of mismatches
    # we use this to calculate a similarity measure between the two agents (referred to as conversion accuracy in the thesis)
    if compare_against is not None:
        mismatches = 0
        correct = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation = env.reset()

        total_reward = 0
        state = torch.tensor(observation, device=device).float()

        for t in count():
            if render:
                if i_episode % 1 == 0:
                    env.render()
            # Select and perform an action
            # we chose a random action with probability epsilon
            if random.random()>=epsilon:
                action = policy_net.forward(state).argmax().detach().item()
                print(policy_net.forward(state))

                # if compare against is not None, we compute the prediction of the compare network
                if compare_against is not None:
                    action_comp = compare_against.forward(state).argmax().detach().item()
                    if action == action_comp:
                        correct += 1
                    else:
                        mismatches += 1
            else:
                action = random.randint(0,env.action_space.n)
            observation, reward, done, info = env.step(action)

            total_reward += reward
            reward = torch.tensor([reward], device=device)
            observation = torch.tensor(observation, device=device).float()

            # Observe new state
            if not done:
                next_state = observation
            else:
                next_state = None

            # Store the transition in memory
            if save_replay:
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                # save replay memory:
                if save_replay:
                    if len(memory) >= memory.capacity:
                        memory.save(policy_net)
                        print('replay memory saved')
                        memory_saved = True
                # plot the rewards
                episode_rewards.append(total_reward)
                plot_durations(episode_rewards, title='', plot_mean=False)
                plot_mean(episode_rewards)
                # if compare_against is not None, report the current Similarity
                if compare_against is not None:
                    print('Similarity (Conversion Accuracy) after ' + str(correct + mismatches) + ' iterations: ' + str(
                        correct * 100 / (mismatches + correct)) + '%')
                break

    print('Complete')
    if save_replay and not memory_saved:
        memory.save(policy_net)
        warnings.warn('save_replay was set to True, but not enough episodes were run to fill the replay.'
                      ' You might want to rerun with a higher number of episodes such that the memory has the correct size')
    print('Mean: ', statistics.mean(episode_rewards))
    print('Std: ', statistics.stdev(episode_rewards))
    # if compare_against is not None, report the final Similarity
    if compare_against is not None:
        print('Similarity (Conversion Accuracy) after ' + str(correct + mismatches) + ' iterations: ' + str(
            correct * 100 / (mismatches + correct)) + '%')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
