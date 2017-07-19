import numpy as np
import sys
import random
import os
from itertools import chain


class GameOptions(object):

    def __init__(self, num_hunters=None, num_rabbits=None, grid_size=None, timestep_reward=None, capture_reward=None,
                 end_when_capture=None):

        self.num_hunters = num_hunters
        self.num_rabbits = num_rabbits
        self.grid_size = grid_size
        self.timestep_reward = timestep_reward
        self.capture_reward = capture_reward
        self.end_when_capture = end_when_capture


class RabbitHunter(object):
    """
        This is a multi-agent learning task, where hunters (agents) are trying to
        catch rabbits in an nxn grid.
        Hunters and rabbits are initialized randomly on the grid.
        An episode ends when all rabbits have been captured.
        There is a reward of +1 reward on capturing a rabbit.
        States are size 3*num_hunters + 3*num_rabbits flattened arrays of:
          concat(hunter states, rabbit states)
        States are of the form:
          [in-game, y-position, x-position], so
          [1, 0, 0] = top-left, [1, 0, n-1] = top-right, [0, -1, -1] = removed
        Actions are size 2*k flattened arrays of:
          concat(hunter 1 movement, hunter 2 movement, ..., hunter k movement)
        Movements are of the form:
          [0, 1] = right, [-1, 1] = up-right, [0, 0] = stay, etc.
          
        The public functions are:
        
        reset(): reset the setting of the environment 
        set_options(options): set options for environment. options should be a GameOptions object.
        start_state(): retrieve a starting state
        perform_action(state, act_indices): perform an action given a state. Return next state and reward. 
        filter_invalid_acts(state, agent_number): determine the invalid actions for an agent given a state. 
        is_end(state): given a state, determine if the game should end.
        render(state): given a state, render the state as a grid (useful for debugging purpose).
        
        A common usage pattern of the environment is:
        
        # Create game options
        game_options = GameOptions(kwargs)
        
        # Create environment
        game = RabbitHunter(game_options)
        
        # Retrieve starting state
        state = game.start_state()
        
        # Run an episode by picking actions and performing action until ending state
        do:
            next_state, reward = game.perform_action(state, act_indices)
        until:
            game.is_end(next_state) 
    """

    action_space = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ]

    agent_rep_size = 3

    def __init__(self, options):
        self.initial_options = options
        self.set_options(options)
        print(f'pid: {os.getpid()}, {options.__dict__}')
        sys.stdout.flush()

    def reset(self):
        self.set_options(self.initial_options)

    def set_options(self, options):
        self.num_hunters = options.num_hunters

        self.num_rabbits = options.num_rabbits

        self.num_agents = self.num_hunters + self.num_rabbits

        self.grid_size = options.grid_size
        self.timestep_reward = options.timestep_reward
        self.capture_reward = options.capture_reward

        self.end_when_capture = options.end_when_capture

    def get_min_state_size(self):
        # 2 because there must be at least one hunter and one rabbit
        return RabbitHunter.agent_rep_size * 2

    def start_state(self):
        """Returns a random initial state. The state vector is a flat array of:
           concat(hunter positions, rabbit positions). Do not allow for overlapping hunters and rabbits."""

        # 1 for active status
        possible_poses = [(1, i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        rabbit_poses = random.choices(possible_poses, k=self.num_rabbits)

        possible_poses = [pos for pos in possible_poses if pos not in rabbit_poses]

        hunter_poses = random.choices(possible_poses, k=self.num_hunters)

        state_as_one_d = list(chain.from_iterable(hunter_poses + rabbit_poses))
        # print('state_as_one_d', state_as_one_d)
        # return RabbitHunter._one_d_to_two_d(state_as_one_d, self)
        return state_as_one_d

    def perform_action(self, state, num_hunters,  a_indices):
        """Performs an action given by a_indices in state s. Returns:
           (s_next, reward)"""
        # state = [[0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0]]
        # print('state', state)
        # state = RabbitHunter._two_d_to_one_d(state, self)
        # print('state', state)

        a = [RabbitHunter.action_space[i] for i in a_indices]
        reward = self.timestep_reward

        # Get positions after hunter and rabbit actions
        # np.zeros(num_hunters * 3, dtype=np.int)
        hunter_poses = []
        for hunter in range(0, num_hunters):
            hunter_idx = hunter * 3
            hunter_act = a[hunter]
            hunter_poses.append([state[hunter_idx + 1] + hunter_act[0], state[hunter_idx + 2] + hunter_act[1]])

        # Must be int here to be valid list index
        rabbit_start_at = int(len(state) / 2)
        active_rabbit_poses = []
        # Assume num_hunters = num_rabbits
        for rabbit in range(0, num_hunters):
            rabbit_idx = rabbit_start_at + rabbit * 3
            r_pos = [state[rabbit_idx + 1], state[rabbit_idx + 2]]
            try:
                hunter_poses.remove(r_pos)
                reward += self.capture_reward
            except ValueError:
                active_rabbit_poses.append(r_pos)

        hunters = [[1] + pos for pos in hunter_poses]
        rabbits = [[1] + pos for pos in active_rabbit_poses]

        one_d = list(chain.from_iterable(hunters + rabbits))

        # return RabbitHunter._one_d_to_two_d(one_d, self), reward
        return one_d, reward

    def _out_of_grid(self, value):
        if value < 0 or value >= self.grid_size:
            return True

    def filter_invalid_acts(self, state, agent_no):
        """Filter the actions available for an agent in a given state. Returns a
           bitmap of available actions (avail action 0, not avail action 1).
           This format is used to speed up masked softmax.
           
           Hunter should be active.
           E.g. an agent in a corner is not allowed to move into a wall."""

        # state = RabbitHunter._two_d_to_one_d(state, self)

        action_size = len(RabbitHunter.action_space)
        avail_a = [0] * action_size
        hunter_pos = state[3 * agent_no + 1:3 * agent_no + 3]

        for i in range(action_size):
            # Check if action moves us off the grid
            a = RabbitHunter.action_space[i]
            new_y = hunter_pos[0] + a[0]
            if self._out_of_grid(new_y):
                avail_a[i] = 1
                continue
            new_x = hunter_pos[1] + a[1]
            if self._out_of_grid(new_x):
                avail_a[i] = 1
        return avail_a

    def is_end(self, state):
        """Given a state, return if the game should end."""

        # state = RabbitHunter._two_d_to_one_d(state, self)

        if len(state) == 0:
            return True
        if self.end_when_capture is not None:
            num_rabbits_remaining = RabbitHunter._get_num_rabbits_from_state_size(len(state))
            if (self.num_rabbits - num_rabbits_remaining) >= self.end_when_capture:
                return True
        return False

    @staticmethod
    def get_num_hunters_from_state_size(state_size):
        return int(state_size / RabbitHunter.agent_rep_size / 2)

    @staticmethod
    def _get_num_rabbits_from_state_size(state_size):
        return int(state_size / RabbitHunter.agent_rep_size / 2)

    @staticmethod
    def _get_hunters_state_from_state(state):
        return state[:RabbitHunter.get_num_hunters_from_state_size(len(state)) * RabbitHunter.agent_rep_size]

    @staticmethod
    def _get_rabbits_state_from_state(state):
        return state[RabbitHunter.get_num_hunters_from_state_size(len(state)) * RabbitHunter.agent_rep_size:]

    @staticmethod
    def get_hunters_from_state(state, game):

        # state = RabbitHunter._two_d_to_one_d(state, game)

        hunters_state = RabbitHunter._get_hunters_state_from_state(state)
        return [hunters_state[i:i + RabbitHunter.agent_rep_size]
                for i in range(0, len(hunters_state), RabbitHunter.agent_rep_size)]

    @staticmethod
    def get_rabbits_from_state(state, game):

        # state = RabbitHunter._two_d_to_one_d(state, game)

        rabbits_state = RabbitHunter._get_rabbits_state_from_state(state)
        return [rabbits_state[i:i + RabbitHunter.agent_rep_size]
                for i in range(0, len(rabbits_state), RabbitHunter.agent_rep_size)]

    @staticmethod
    def _get_poses_from_one_d_array(array):
        positions = []
        for idx in range(0, len(array), 3):
            # +1 to skip the status number
            positions.append(array[idx+1: idx+3])
        return positions

    def render(self, state, game, outfile=sys.stdout):

        # state = RabbitHunter._two_d_to_one_d(state, self)

        num_hunter = self.get_num_hunters_from_state_size(len(state))

        hunter_poses = RabbitHunter._get_poses_from_one_d_array(
            state[:num_hunter * RabbitHunter.agent_rep_size])

        rabbit_poses = RabbitHunter._get_poses_from_one_d_array(
            state[num_hunter * RabbitHunter.agent_rep_size:])

        outfile.write(f'Rendering state: {state}\n')

        for row in range(self.grid_size):
            draw = ''
            for col in range(self.grid_size):
                pos = [row, col]

                num_h_here = hunter_poses.count(pos)
                for _ in range(num_h_here):
                    draw += 'h'

                num_r_here = rabbit_poses.count(pos)
                for _ in range(num_r_here):
                    draw += 'r'

                if num_h_here is 0 and num_r_here is 0:
                    draw += '_'

                draw += '\t'

            draw += '\n\n'
            outfile.write(draw)

        outfile.write('\n')

    @staticmethod
    def _one_d_to_two_d(one_d, game, should_assert=False):
        # print('one_d', one_d)
        two_d = [[0 for _ in range(game.grid_size)] for _ in range(game.grid_size)]
        # print('two_d', two_d)
        # num_hunters = RabbitHunter.get_num_hunters_from_state_size(len(one_d))
        num_hunters = 1

        for nh in range(num_hunters):
            hunter_idx = nh * 3
            h_y = one_d[hunter_idx + 1]
            h_x = one_d[hunter_idx + 2]
            two_d[h_y][h_x] = 1

        # Assume num hunters = num rabbits
        rabbit_start_at = int(len(one_d) / 2)

        for nr in range(num_hunters):
            rabbit_idx = rabbit_start_at + nr * 3
            r_y = one_d[rabbit_idx + 1]
            r_x = one_d[rabbit_idx + 2]
            two_d[r_y][r_x] = -1

        # if should_assert:
        #     assert RabbitHunter._two_d_to_one_d(two_d, game, should_assert=True) == one_d

        return two_d

    @staticmethod
    def _two_d_to_one_d(two_d, game, should_assert=False):
        # TODO: decide on how to keep the order of the hunter
        # TODO: make this function works for more than one hunter - rabbit pair

        one_d = []

        h_found = False
        r_found = False

        for row_idx, row in enumerate(two_d):
            if h_found and r_found:
                break

            if not h_found:
                try:
                    h_x = row.index(1)
                    one_d = [1, row_idx, h_x] + one_d
                    h_found = True
                except ValueError:
                    pass

            if not r_found:
                try:
                    r_x = row.index(-1)
                    one_d += [1, row_idx, r_x]
                    r_found = True
                except ValueError:
                    pass

        # if should_assert:
        #     assert RabbitHunter._one_d_to_two_d(one_d, game, should_assert=True) == two_d

        return one_d


def array_equal(a, b):
    """Because np.array_equal() is too slow. Three-element arrays only."""
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

# def valid_state(s):
#     """Returns if the given state vector is valid."""
#     return s.shape == (3*k+3*m, ) and \
#            np.all([-1 <= e < n for e in s]) and \
#            np.all([e in (0, 1) for e in s[::3]])

# def valid_action(a):
#     """Returns if the given action vector is valid"""
#     return a.shape == (2*k, ) and np.all([-1 <= e <= 1 for e in a])

