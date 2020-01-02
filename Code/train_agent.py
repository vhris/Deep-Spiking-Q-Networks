import torch
import torch.optim as optim
import random
import os
import torch.nn.functional as F
import gym
from collections import namedtuple
from itertools import count
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .replay_memory import ReplayMemory
from .utils import plot_durations, save_model,save_hyperparameters



def select_action(policy_net,state,eps_start,eps_end,eps_decay,n_actions,device,steps_done):
    """Selects the action with the highest Q value with probability 1-eps_threshold and else a random action
    Args:
        policy_net: the net which determines the action
        state: the current state of the environment
        eps_start: the initial value of epsilon
        eps_end: The minimum value of epsilon
        eps_decay: The decay of epsilon
        n_actions: number of possible actions
        device: device for the torch tensors
        steps_done: number of steps done so far to determine the decay
    Returns: The selected action"""
    sample = random.random()
    eps_threshold = max(eps_end, eps_start * (eps_decay ** steps_done))
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.tensor([[policy_net.forward(state).argmax()]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory,BATCH_SIZE,device,policy_net, Transition, n_actions, target_net,gamma,optimizer,double_q_learning=True,
                   gradient_clipping=True,initial_replay_size=0):
    """Does one update of the optimizer for the policy_net
    Args:
        memory: The reply memory from which transitions are sampled
        BATCH_SIZE: the size of the sample batch
        device: The device for the torch tensors
        Transition: the transition tuple in use
        n_actions: the number of possible actions
        target_net: the current target net
        gamma: hyperparameter of the DQN
        optimizer: the optimizer used
        double_q_learning: Whether to use double or "normal" Q-Learning
        gradient_clipping: Whether to use gradient clipping or not
        stack_or_cat: This is a workaround. If the input state has only one dimension the tensors have to be stacked, else they have to be concatenated
    Returns: the updated optimizer and policy_net or None, if the memory is smaller than the batch_size"""
    if len(memory) < BATCH_SIZE or len(memory)<initial_replay_size:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # use stack if the input state has only one dimension (is a vector)
    #TODO test this
    if batch.state[0].dim() == 1:
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.stack(batch.state)
    # use this if the input state has several dimensions
    else:
        non_final_next_states = torch.cat([s for s in batch.next_state
                                             if s is not None])
        state_batch = torch.cat(batch.state)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net.forward(state_batch).gather(1, action_batch)

    if not double_q_learning:
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net.forward(non_final_next_states).max(1)[0].detach()
    else:
        #Double Q Learning
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #We use Double Q Learning, that is the best action is chosen with the policy net, but it's value is calculated with the target net
        next_state_best_actions = torch.zeros([BATCH_SIZE,n_actions], device=device)
        next_state_best_actions[non_final_mask] = policy_net.forward(non_final_next_states)

        best_action_mask = torch.zeros_like(next_state_best_actions,dtype=torch.bool,device=device)
        best_action_mask[torch.arange(len(next_state_best_actions)), next_state_best_actions.argmax(1)] = True

        next_state_values[non_final_mask] = target_net.forward(non_final_next_states).masked_select(
            best_action_mask[non_final_mask].bool())

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute mse loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()

    loss.backward()
    if gradient_clipping:
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return optimizer, policy_net


def train_agent(environment, policy_net,target_net, batch_size,gamma,eps_start,eps_end,eps_decay,target_update,optimizer,learning_rate,
                memory_size,device, gym_target_average,gym_target_stay,num_episodes=1000,max_steps=None, render =True,
                double_q_learning=True, gradient_clipping=True,initial_replay_size=0,gym_seed=None, torch_seed=None,random_seed = None):
    """Args:
            environment: the name of the gym environment as a string
            policy_net: the policy network for the agent
            target_net: the target network for the agent which should be a copy of the policy network initially
            batch_size: size of one training batch
            gamma: discount factor
            eps_start: initial epsilon value
            eps_end: minimum epsilon value
            eps_decay: linear decay factor
            target_update: update frequency of the target net
            optimizer: the optimizer that is used
            learning_rate: learning rate of the optimizer
            memory_size: size of the replay memory
            device: device for torch tensors
            gym_target_average: 100 episode average goal for the environment
            gym_target_stay: how long the agent needs to stay at the target_average to solve the environment
            num_episodes: number of training episodes
            max_steps: time limit for the environment, if None then no time limit
            render: if the environment should be rendered
            double_q_learning: whether to use double Q Learning
            gradient_clipping: whether to use gradient clipping
            initial_replay_size: at what memory size is the training started
            gym_seed: manual seed for the gym environment (optional)
            """
    # set up environment
    env = gym.make(environment)
    # set seeds if specified
    if gym_seed is not None:
        env.seed(gym_seed)
        # this line is necessary to reproduce our results, because it was included in the original script. Else it could be left out.
        env.reset()
    # set maximum episode steps
    if max_steps is not None:
        env._max_episode_steps = max_steps
    else:
        max_steps = np.inf
        env._max_episode_steps = np.inf
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # define transition tuple
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    memory = ReplayMemory(memory_size)

    # some helper variables
    steps_done = 0
    episode_rewards = []
    # average counter counts how long the agent is already above the gym target
    avg_counter = 0
    # best average keeps track of the best average and best_average_after at what episode it was reached
    best_average = 0
    best_average_after = np.inf
    # finished tells if the gym standard is reached and finished_after at what episode
    finished = False
    finished_after = np.inf

    # save initial model
    save_model(policy_net, 'initial')
    save_hyperparameters(gamma,learning_rate,eps_start,eps_end,eps_decay,memory_size,batch_size,target_update,
                         others='Gym seed: '+str(gym_seed)+' , Torch seed: '+str(torch_seed)+' , Random seed: '+str(random_seed))

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation = env.reset()
        state = torch.tensor(observation, device=device).float()

        total_reward = 0
        for t in count():
            if render:
                env.render()

            # Select and perform an action
            action = select_action(policy_net, state, eps_start, eps_end, eps_decay, n_actions, device, steps_done)
            steps_done += 1
            observation, reward, done, info = env.step(action.item())

            total_reward += reward
            reward = torch.tensor([reward], device=device)

            observation = torch.tensor(observation, device=device).float()

            # Observe new state
            if not done:
                next_state = observation
            else:
                next_state = None

            # Store the transition in memory, but only if the maximum number of steps has not been reached.
            # If the last transition is stored in memory it can distort the results, because it always appears to be a bad state (no further reward)
            if t < max_steps-1:
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            opt = optimize_model(memory, batch_size, device, policy_net, Transition, n_actions, target_net, gamma,
                                 optimizer,double_q_learning=double_q_learning,gradient_clipping=gradient_clipping,initial_replay_size=initial_replay_size)
            if opt is not None:
                optimizer, policy_net = opt
            if done:
                rewards_t = torch.tensor(episode_rewards, dtype=torch.float)

                episode_rewards.append(total_reward)
                plot_durations(episode_rewards)
                if len(rewards_t) >= 100:
                    # plot the episode durations
                    average = rewards_t[rewards_t.shape[0] - 100:rewards_t.shape[0]].mean()
                    #print(average, avg_counter)
                    if average > best_average:
                        best_average = average
                        best_average_after = i_episode
                    if average >= gym_target_average:
                        avg_counter += 1
                    else:
                        avg_counter = 0
                    # save neural network if open ai gym standard is reached:
                    if (avg_counter >= gym_target_stay) and not finished:
                        save_model(policy_net, 'trained')
                        # break the training loop
                        finished = True
                        finished_after = i_episode
                #stop the episode by break
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # print some information about the training process
    if finished:
        print('OpenAIGymStandard reached at episode ', finished_after)
    else:
        print('Failed to reach OpenAIGymStandard')
    print('Best average: ', best_average, ' reached at episode ', best_average_after)
    print('Complete')
    env.close()
    plt.savefig('Training.png')
    plt.ioff()
    plt.show()
