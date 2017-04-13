'''
    This is a multi-agent SARSA implementation for any game that conforms to
    the interface:

      game.num_agents - number of agents
      game.start_state() - returns start state of the game
      game.is_end(state) - given a state, return if the game/episode has ended
      game.perform_action(s, a) - given action indices at state s, returns next_s, reward
      game.filter_actions(s, n) - filter actions available for an agent in a given state
      game.set_options(options) - set options for the game
'''

import argparse
import numpy as np
import os
import random
import sys
import torch
import time

from namedlist import namedlist
from torch.autograd import Variable

# Define a EpisodeStep container for each step in an episode:
#   s, a is the state-action pair visited during that step
#   r is the reward received from that state-action pair
#   G is the discounted return received from that state-action pair
EpisodeStep = namedlist('EpisodeStep', 's a r G', default=0)
SMALL = 1e-7

def run_episode(policy_net, gamma=1.0):
    '''Runs one episode of a game to completion with a policy network,
       which is a LSTM that maps states to action probabilities.

       Parameters:
           policy_net: LSTM policy network
           gamma: discount factor for calculating returns

       Returns:
           [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
    # Initialize state as player position
    state = game.start_state()
    episode = []

    # Run game until agent reaches the end
    while not game.is_end(state):
        # Let our agent decide that to do at this state
        a_indices = run_policy_net(policy_net, state)

        # Take that action, then the game gives us the next state and reward
        next_s, r = game.perform_action(state, a_indices)

        # Record state, action, reward
        episode.append(EpisodeStep(s=state, a=a_indices, r=r))
        state = next_s

        # Terminate episode early
        if len(episode) > max_episode_len:
            episode[-1].r += max_len_penalty
            break

    # We have the reward from each (state, action), now calculate the return
    for i, step in enumerate(reversed(episode)):
        if i == 0: step.G = step.r
        else: step.G = step.r + gamma*episode[len(episode)-i].G

    return episode

def build_policy_net(layers):
    '''Builds an LSTM policy network, which maps states to action vectors.

       More precisely, the input into the LSTM will be a vector consisting of
       [prev_output, state]. The output of the LSTM will be a vector that
       gives unnormalized probabilities of each action for the agents; softmax
       is applied afterwards, see run_policy_net(). This model only handles one
       time step, i.e. one agent, so it must be manually re-run for each agent.
    '''
    class PolicyNet(torch.nn.Module):
        def __init__(self, layers):
            super(PolicyNet, self).__init__()
            self.lstm = torch.nn.LSTMCell(layers[0], layers[1])
            self.linear = torch.nn.Linear(layers[1], layers[2])
            self.layers = layers

        def forward(self, x, h0, c0):
            h1, c1 = self.lstm(x, (h0, c0))
            o1 = self.linear(h1)
            return o1, h1, c1

    policy_net = PolicyNet(layers)
    return policy_net.cuda() if cuda else policy_net

def masked_softmax(logits, mask):
    """
    Parameters:
        logits: Variable of size [batch_size, d]
        mask: FloatTensor of size [batch_size, d]

    Returns:
        probs: row-wise masked softmax of the logits
    """
    scores = torch.exp(logits) * Variable(mask)
    partitions = torch.sum(scores, 1)
    probs = scores / partitions.expand_as(scores)
    return probs

def run_policy_net(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return an action vector, as well as parameter gradients.

       Essentially, run the policy_net LSTM for game.num_agents time-steps, to
       get the action for each agent, conditioned on previous agents' actions.
       As such, the input to the LSTM at time-step n is concat(a_{n-1}, state).
       We return a list of action indices, one index per agent.

       Parameters:
           policy_net: LSTM that given (x_n, h_n, c_n) returns o_nn, h_nn, c_nn
           state: state of the MDP

       Returns:
           a_indices: list of action indices of each agent
    '''
    # Prepare initial inputs for policy_net
    a_indices = []
    h_size, a_size = policy_net.layers[1], policy_net.layers[2]
    a_n = np.zeros(a_size)
    h_n, c_n = Variable(ZeroTensor(1, h_size)), Variable(ZeroTensor(1, h_size))
    x_n = Variable(FloatTensor([np.append(a_n, state)]))
    policy_net.zero_grad()

    # Use policy_net to predict output for each agent
    for n in range(game.num_agents):
        # Do a forward step through policy_net, filter actions, and softmax it
        x_n.data.copy_(torch.Tensor([np.append(a_n, state)]))
        o_n, h_n, c_n = policy_net(x_n, h_n, c_n)

        # Select action over possible ones
        action_mask = FloatTensor(game.filter_actions(state, n)).unsqueeze(0)
        dist = masked_softmax(o_n, action_mask)
        a_index = torch.multinomial(dist.data,1)[0,0]

        # Record action for this iteration/agent
        a_indices.append(a_index)

        # Prepare inputs for next iteration/agent
        a_n = np.zeros(a_size)
        a_n[a_index] = 1

    return a_indices

def train_policy_net(policy_net, episode, gamma=1.0):
    '''Update the policy network parameters with a SARSA update.

       That is, we want the policy net to minimize the squared error:
         (r + gamma*Q(s', a') - Q(s, a))^2
       So, update the parameters using the gradient of the error.

       Parameters:
           policy_net: LSTM policy network
           episode: list of EpisodeStep's
           gamma: discount term
    '''
    # Prepare for one forward pass, with the batch containing the entire episode
    a_indices = []
    h_size, a_size = policy_net.layers[1], policy_net.layers[2]
    a_n = np.zeros(a_size)
    h_n_batch = Variable(ZeroTensor(len(episode), h_size))
    c_n_batch = Variable(ZeroTensor(len(episode), h_size))
    policy_net.zero_grad()

    # Input to LSTM has size [num_agents, episode_len, each_input_size]
    input_batch = ZeroTensor(game.num_agents, len(episode),
                             a_size + len(episode[0].s))

    # Fill input_batch with concat(a_{n-1}, state) for each agent, for each time-step
    for i in range(game.num_agents):
        for j, step in enumerate(episode):
            input_batch[i, j, a_size:].copy_(torch.Tensor(step.s))
            if i > 0: input_batch[i, j, step.a[i-1]] = 1
    input_batch = Variable(input_batch)

    # Fill action_mask_batch with action masks for each state
    action_mask_batch = ZeroTensor(game.num_agents, len(episode), a_size)
    for i in range(game.num_agents):
        for j, step in enumerate(episode):
            action_mask_batch[i,j,:].copy_(torch.Tensor(game.filter_actions(step.s, i)))

    # Do a forward pass, and fill sum_log_probs with sum(log(p)) for each time-step
    sum_log_probs = Variable(ZeroTensor(len(episode)))
    for i in range(game.num_agents):
        o_n, h_n_batch, c_n_batch = policy_net(input_batch[i], h_n_batch, c_n_batch)
        dist = masked_softmax(o_n, action_mask_batch[i])
        for j, step in enumerate(episode):
            sum_log_probs[j] = sum_log_probs[j] + torch.log(dist[j, step.a[i]])

    # Do a backward pass to compute the policy gradient term
    R = Variable(FloatTensor(np.asarray([step.r for step in episode])))
    Q_next = Variable(torch.cat((sum_log_probs.data[1:], ZeroTensor(1))))
    error = ((R + gamma*Q_next - sum_log_probs)**2).mean()
    error.backward()

    # Clip gradients to [-1, 1]
    for W in policy_net.parameters():
        W.grad.data.clamp_(-1,1)

    # Do a step of RMSProp
    optimizer_policy_net.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs multi-agent policy gradient.')
    parser.add_argument('--game', choices=['gridworld', 'gridworld_3d', 'hunters'], required=True, help='A game to run')
    parser.add_argument('--cuda', default=False, action='store_true', help='Include to run on CUDA')
    parser.add_argument('--max_episode_len', default=float('inf'), type=float, help='Terminate episode early at this number of steps')
    parser.add_argument('--max_len_penalty', default=0, type=float, help='If episode is terminated early, add this to the last reward')
    parser.add_argument('--num_episodes', default=100000, type=int, help='Number of episodes to run in a round of training')
    parser.add_argument('--num_rounds', default=1, type=int, help='How many rounds of training to run')
    parser.add_argument('--gamma', default=1, type=float, help='Global discount factor for Monte-Carlo returns')
    args = parser.parse_args()
    print(args)

    # Sets options for PG
    cuda = args.cuda
    max_episode_len = args.max_episode_len
    max_len_penalty = args.max_len_penalty
    if cuda: print('Running policy gradient on GPU.')

    # Transparently set number of threads based on environment variables
    num_threads = int(os.getenv('OMP_NUM_THREADS', 1))
    torch.set_num_threads(num_threads)

    # Define wrappers for Tensors
    FloatTensor = lambda x: torch.cuda.FloatTensor(x) if cuda else torch.FloatTensor(x)
    ZeroTensor = lambda *s: torch.cuda.FloatTensor(*s).zero_() if cuda else torch.zeros(*s)
    ByteTensor = lambda x: torch.cuda.ByteTensor(x) if cuda else torch.ByteTensor(x)

    if args.game == 'gridworld':
        import gridworld as game
        policy_net_layers = [5, 32, 3]
        game.set_options({'grid_y': 4, 'grid_x': 4})
    elif args.game == 'gridworld_3d':
        import gridworld_3d as game
        policy_net_layers = [6, 32, 3]
        game.set_options({'grid_z': 6, 'grid_y': 6, 'grid_x': 6})
    elif args.game == 'hunters':
        import hunters as game
        policy_net_layers = [17, 128, 9]
        game.set_options({'rabbit_action': None, 'remove_hunters': True,
                          'capture_reward': 10})

    for i in range(args.num_rounds):
        policy_net = build_policy_net(policy_net_layers)
        optimizer_policy_net = torch.optim.RMSprop(policy_net.parameters(), lr=1e-3, eps=1e-5)

        avg_value_error, avg_return = 0.0, 0.0
        for num_episode in range(args.num_episodes):
            #t = time.time()
            episode = run_episode(policy_net, gamma=args.gamma)
            #time_episode = time.time() - t
            avg_return = 0.9 * avg_return + 0.1 * episode[0].G
            print("{{'i': {}, 'num_episode': {}, 'episode_len': {}, 'episode_return': {}, 'avg_return': {}}},".format(i, num_episode, len(episode), episode[0].G, avg_return))
            #print time_episode
            #t = time.time()
            train_policy_net(policy_net, episode, gamma=args.gamma)
            #print time.time() - t