"""
    This is a multi-agent learning task, where k hunters (agents) are trying to
    catch m rabbits in an nxn grid.

    Hunters and rabbits are initialized randomly on the grid, with overlaps.
    An episode ends when all rabbits have been captured. Rabbits can have
    different movement patterns. There is a reward of -1 per time step (and
    optionally a +1 reward on capturing a rabbit).

    States are size 3*k+3*m flattened arrays of:
      concat(hunter positions, rabbit positions)
    Positions are of the form:
      [in-game, y-position, x-position], so
      [1, 0, 0] = top-left, [1, 0, n-1] = top-right, [0, -1, -1] = removed

    Actions are size 2*k flattened arrays of:
      concat(hunter 1 movement, hunter 2 movement, ..., hunter k movement)
    Movements are of the form:
      [0, 1] = right, [-1, 1] = up-right, [0, 0] = stay, etc
"""

import numpy as np
import sys

outfile = sys.stdout

class GameOptions(object):

    def __init__(self, num_hunters=None, num_rabbits=None, grid_size=None, timestep_reward=None, capture_reward=None):

        self.num_hunters = num_hunters
        self.num_rabbits = num_rabbits
        self.grid_size = grid_size
        self.timestep_reward = timestep_reward
        self.capture_reward = capture_reward


class RabbitHunter(object):

    action_space = [
        np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
        np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
        np.array([1, -1]), np.array([1, 0]), np.array([1, 1])
    ]

    num_agent_type = 2

    def __init__(self, options):
        self.initial_options = options
        self.set_options(options)

        self.agent_rep_size = 2
        print(options.__dict__)

    def reset(self):
        self.set_options(self.initial_options)

    def set_options(self, options):
        self.num_hunters = options.num_hunters
        self.num_active_hunters = options.num_hunters

        self.num_rabbits = options.num_rabbits
        self.num_active_rabbits = options.num_rabbits

        self.num_agents = self.num_hunters + self.num_rabbits
        self.num_active_agents = self.num_active_hunters + self.num_active_rabbits

        self.grid_size = options.grid_size
        self.timestep_reward = options.timestep_reward
        self.capture_reward = options.capture_reward

    def get_min_state_size(self):
        return int(self.agent_rep_size * RabbitHunter.num_agent_type)

    def start_state(self):
        '''Returns a random initial state. The state vector is a flat array of:
           concat(hunter positions, rabbit positions).'''

        state_size = self.agent_rep_size * (self.num_hunters + self.num_rabbits)

        start = np.random.randint(0, self.grid_size, size=state_size)

        return start

    def perform_action(self, state, a_indices):
        '''Performs an action given by a_indices in state s. Returns:
           (s_next, reward)'''
        a = action_indices_to_coordinates(a_indices)
        assert self.valid_action(a)
        assert self.valid_state(state)
        # print()
        # print('inside perform action')

        # Get positions after hunter actions
        hunter_pos = np.zeros(self.num_active_hunters * self.agent_rep_size, dtype=np.int)
        # print('empty hunter_pos', hunter_pos)
        for hunter in range(0, self.num_active_hunters):
            hunter_idx = hunter * self.agent_rep_size

            hunter_act = a[hunter_idx:hunter_idx + 2]
            sa = state[hunter_idx:hunter_idx + self.agent_rep_size] + hunter_act

            hunter_pos[hunter_idx:hunter_idx + self.agent_rep_size] = np.clip(sa, 0, self.grid_size - 1)

        # Remove rabbits (and optionally hunters) that overlap
        reward = self.timestep_reward
        rabbit_pos = state[self.num_active_hunters * self.agent_rep_size:]

        # print('state before', state)
        # print('a_indices', a_indices)
        # print('hunter position', hunter_pos)
        # print('rabbit position', rabbit_pos)
        captured_rabbit_idxes = []
        inactive_hunter_idxes = []
        for hunter in range(0, self.num_active_hunters):
            h_idx = hunter * self.agent_rep_size

            for rabbit in range(0, self.num_active_rabbits):
                r_idx = rabbit * self.agent_rep_size

                h_pos = hunter_pos[h_idx:h_idx + self.agent_rep_size]
                r_pos = rabbit_pos[r_idx:r_idx + self.agent_rep_size]

                if array_equal(h_pos, r_pos) and active_agent(h_pos) and active_agent(r_pos):
                    # print(f'hunter {h_pos} catches rabbit {r_pos}')

                    hunter_pos[h_idx:h_idx + self.agent_rep_size] = [-1, -1]
                    rabbit_pos[r_idx:r_idx + self.agent_rep_size] = [-1, -1]

                    captured_rabbit_idxes += [r_idx, r_idx + 1]
                    inactive_hunter_idxes += [h_idx, h_idx + 1]

                    reward += self.capture_reward
                    self.num_active_hunters -= 1
                    self.num_active_rabbits -= 1
                    self.num_active_agents -= 2

        # print('captured_rabbit_idxes', captured_rabbit_idxes)
        # print('inactive_hunter_idxes', inactive_hunter_idxes)
        #
        # print('before delete')
        # print('hunter position', hunter_pos)
        # print('rabbit position', rabbit_pos)

        rabbit_pos = np.delete(rabbit_pos, captured_rabbit_idxes, axis=0)
        hunter_pos = np.delete(hunter_pos, inactive_hunter_idxes, axis=0)

        # print('after delete')
        # print('hunter position', hunter_pos)
        # print('rabbit position', rabbit_pos)
        #
        # Return (s_next, reward)
        s_next = np.concatenate((hunter_pos, rabbit_pos))

        # print('s_next', s_next)
        # sys.stdout.flush()

        return s_next, reward

    def filter_actions(self, state, agent_no):
        '''Filter the actions available for an agent in a given state. Returns a
           bitmap of available actions. Hunter should be active.
           E.g. an agent in a corner is not allowed to move into a wall.'''
        avail_a = np.ones(len(RabbitHunter.action_space), dtype=int)

        hunter_pos = state[self.agent_rep_size * agent_no:self.agent_rep_size * agent_no + self.agent_rep_size]

        for i in range(len(RabbitHunter.action_space)):
            # Check if action moves us off the grid
            a = RabbitHunter.action_space[i]
            sa = hunter_pos + a
            if (sa[0] < 0 or sa[0] >= self.grid_size) or (sa[1] < 0 or sa[1] >= self.grid_size):
                avail_a[i] = 0
        return avail_a

    def is_end(self, state):
        '''Given a state, return if the game should end.'''
        if len(state) == 0:
            return True
        return False

    def get_num_hunters_from_state_size(self, state_size):
        # / 2 because there are two type of agents
        return int(state_size / self.agent_rep_size / RabbitHunter.num_agent_type)

    def valid_state(self, state):
        '''Returns if the given state vector is valid.'''
        return state.shape == (self.agent_rep_size * self.num_active_agents, ) and \
            np.all([0 <= e < self.grid_size for e in state])

    def valid_action(self, a):
        return a.shape == (self.num_active_hunters * len(RabbitHunter.action_space[0]), ) and \
            np.all([-1 <= e <= 1 for e in a])

def action_indices_to_coordinates(a_indices):
    '''Converts a list of action indices to action coordinates.'''
    coords = [RabbitHunter.action_space[i] for i in a_indices]
    return np.concatenate(coords)

def array_equal(a, b):
    '''Because np.array_equal() is too slow. Three-element arrays only.'''
    return a[0] == b[0] and a[1] == b[1]

def active_agent(position):
    if -1 in position:
        return False
    return True


# def valid_action(a):
#     '''Returns if the given action vector is valid'''
#     return a.shape == (2*k, ) and np.all([-1 <= e <= 1 for e in a])

