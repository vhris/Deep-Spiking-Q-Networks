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

def load_agent(environment,policy_net, device,epsilon=0,gym_seed=None,save_replay=False,replay_size=2*10**4,
               max_steps=None,num_episodes=100,render =True, compare_against=None,
               input_preprocessing=None, observation_history_length=None,frameskip=None,
               no_op_range=None,no_op=None):
    """load an agent on a gym environment

    Args:
        environment: string: name of gym environment
        policy_net: PyTorch neural network or SQN. The agent that determines the actions to take.
                    Potentially other networks can be used, only requirement is that it implements the method forward.
        device: torch device
        epsilon: random action probability
        gym_seed: seed for the environment
        save_replay: whether to save the observed states and their corresponding actions
        replay_size: size of the replay memory
        max_steps: maximum number of steps in one episode. If None no maxmium is defined.
        num_episodes: number of episodes to simulate the environment for
        render: whether to render the environment
        compare_against: None or PyTorch neural network or SQN.
                         If not None, the policy_net is compared against this agent and a similarity measure is computed
        input_preprocessing: optional function depending on the observation history which preprocesses the input before
                             it is passed to the agent(s)
        observation_history_length: None or int. Whether to keep a history of observations and how long it is.
        frameskip: used for Atari environments. Determines the frameskip.
        no_op_range: tuple (min,max). used for Atari environments (see Mnih et al.).
        no_op: "do nothing" action for the game (int)."""

    # set up environment
    env = gym.make(environment)
    if gym_seed is not None:
        env.seed(gym_seed)
    if max_steps is not None:
        env._max_episode_steps = max_steps
    # set up frameskip
    if frameskip is not None:
        env._frameskip = frameskip

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
        # Keep track of the reward in this episode
        total_reward = 0

        # Initialize observation history if needed:
        if observation_history_length is not None:
            observation_history = []
            observation_history.insert(0, observation)

        # compute random number of do nothing operations, if specified:
        if no_op_range is not None:
            # random number or no ops between no_op[0] and no_op[1]
            rand = random.randint(no_op_range[0], no_op_range[1])
            for i in range(0, rand):
                observation, reward, _, _ = env.step(no_op)
                total_reward += reward
                observation_history.insert(0, observation)
                if len(observation_history) > observation_history_length:
                    observation_history.pop(-1)

        # preprocess the input if preprocessing specified
        if input_preprocessing is not None:
            observation = input_preprocessing(observation_history)
        # else the observation needs to be cast to a float tensor for the computation of the neural network
        else:
            observation = torch.tensor(observation, device=device).float()
        state = observation

        for t in count():
            if render:
                if i_episode % 1 == 0:
                    env.render()
            # Select and perform an action
            # we chose a random action with probability epsilon
            if random.random()>=epsilon:
                action = policy_net.forward(state).argmax().detach().item()

                # if compare against is not None, we compute the prediction of the compare network
                if compare_against is not None:
                    action_comp = compare_against.forward(state).argmax().detach().item()
                    if action == action_comp:
                        correct += 1
                    else:
                        mismatches += 1
            else:
                action = random.randint(0,env.action_space.n-1)
            observation, reward, done, info = env.step(action)

            total_reward += reward
            reward = torch.tensor([reward], device=device)

            # if observation history is required, add observation to history
            if observation_history_length is not None:
                observation_history.insert(0, observation)
                if len(observation_history) > observation_history_length:
                    observation_history.pop(-1)

            # preprocess the input if a preprocessing function is specified
            if input_preprocessing is not None:
                observation = input_preprocessing(observation_history)
            # else the observation needs to be cast to a float tensor for the computation of the neural network
            else:
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
                if save_replay and not memory_saved:
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
    env.close()
    plt.ioff()
    plt.show()
