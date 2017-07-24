import numpy as np
import sys
import random
import os
from itertools import chain


def deepcopy_2d_list(original):
    new = []

    for row in original:
        new.append([])
        new[-1] = [num for num in row]

    return new


class GameOptions(object):

    def __init__(self, num_hunters=None, num_rabbits=None, grid_size=None, timestep_reward=None, capture_reward=None,
                 end_when_capture_num_rabbits=None):

        self.num_hunters = num_hunters
        self.num_rabbits = num_rabbits
        self.grid_size = grid_size
        self.timestep_reward = timestep_reward
        self.capture_reward = capture_reward
        self.end_when_capture_num_rabbits = end_when_capture_num_rabbits


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
        self.possible_poses_no_status = [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)]
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

        self.end_when_capture_num_rabbits = options.end_when_capture_num_rabbits

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

        return list(chain.from_iterable(hunter_poses + rabbit_poses))

    def perform_action(self, state, num_hunters,  a_indices):
        """Performs an action given by a_indices in state s. Returns:
           (s_next, reward)"""
        a = [RabbitHunter.action_space[i] for i in a_indices]
        reward = self.timestep_reward

        num_rabbits_captured_ts = 0

        # Get positions after hunter actions
        hunter_poses = []
        for hunter in range(0, num_hunters):
            hunter_idx = hunter * 3
            hunter_act = a[hunter]
            hunter_poses.append([state[hunter_idx + 1] + hunter_act[0], state[hunter_idx + 2] + hunter_act[1]])

        active_hunter_poses = deepcopy_2d_list(hunter_poses)
        # Keep track of hunters and their index to replace inactive hunters later
        # Each item is [idx in original hunter_poses, [y, x]]
        replacing_hunters = []

        # Must be int here to be valid list index
        rabbit_start_at = int(len(state) / 2)
        rabbit_poses = []
        for rabbit in range(0, num_hunters):
            rabbit_idx = rabbit_start_at + rabbit * 3
            rabbit_poses.append([state[rabbit_idx + 1], state[rabbit_idx + 2]])

        rabbit_poses_remaining = deepcopy_2d_list(rabbit_poses)

        active_rabbit_poses = []
        for r_pos in rabbit_poses:
            try:
                remove_hunter_pos_idx = hunter_poses.index(r_pos)
                del hunter_poses[remove_hunter_pos_idx]

                assert r_pos in rabbit_poses_remaining
                rabbit_poses_remaining.remove(r_pos)
                reward += self.capture_reward
                num_rabbits_captured_ts += 1

                # generate new hunters and rabbits and append them to respective lists
                new_hunter = self._generate_agent_pos(rabbit_poses_remaining)
                replacing_hunters.append([remove_hunter_pos_idx, new_hunter])

                new_rabbit = self._generate_agent_pos(hunter_poses)
                active_rabbit_poses.append(new_rabbit)
            except ValueError:
                active_rabbit_poses.append(r_pos)

        # Replace inactive hunters with new hunters
        for new_hunters in replacing_hunters:
            idx = new_hunters[0]
            pos = new_hunters[1]

            active_hunter_poses[idx] = pos

        hunters = [[1] + pos for pos in active_hunter_poses]
        rabbits = [[1] + pos for pos in active_rabbit_poses]

        return list(chain.from_iterable(hunters + rabbits)), reward, num_rabbits_captured_ts

    def _generate_agent_pos(self, can_not_overlap_these):

        possible_poses = [pos for pos in self.possible_poses_no_status if pos not in can_not_overlap_these]

        return random.choice(possible_poses)

    def _out_of_grid(self, value):
        if value < 0 or value >= self.grid_size:
            return True

    def filter_invalid_acts(self, state, agent_no):
        """Filter the actions available for an agent in a given state. Returns a
           bitmap of available actions (avail action 0, not avail action 1).
           This format is used to speed up masked softmax.
           
           Hunter should be active.
           E.g. an agent in a corner is not allowed to move into a wall."""
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

    def is_end(self, num_rabbits_captured):
        """Given a state, return if the game should end."""
        if num_rabbits_captured >= self.end_when_capture_num_rabbits:
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
    def get_hunters_from_state(state):
        hunters_state = RabbitHunter._get_hunters_state_from_state(state)
        return [hunters_state[i:i + RabbitHunter.agent_rep_size]
                for i in range(0, len(hunters_state), RabbitHunter.agent_rep_size)]

    @staticmethod
    def get_rabbits_from_state(state):
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

    def render(self, state, outfile=sys.stdout):
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

