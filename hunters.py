import numpy as np
import sys
import random
import os


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
        filter_actions(state, agent_number): determine the possible actions for an agent given a state. 
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
        np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
        np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
        np.array([1, -1]), np.array([1, 0]), np.array([1, 1])
    ]

    def __init__(self, options):
        self.initial_options = options
        self.set_options(options)
        self.agent_rep_size = 3
        print(f'pid: {os.getpid()}, {options.__dict__}')
        sys.stdout.flush()

    def reset(self):
        self.set_options(self.initial_options)

    def set_options(self, options):
        self.num_hunters = options.num_hunters
        self.num_active_hunters = options.num_hunters

        self.num_rabbits = options.num_rabbits
        self.num_active_rabbits = options.num_rabbits

        self.num_agents = self.num_hunters + self.num_rabbits

        self.grid_size = options.grid_size
        self.timestep_reward = options.timestep_reward
        self.capture_reward = options.capture_reward

        self.end_when_capture = options.end_when_capture

    def get_min_state_size(self):
        # 2 because there must be at least one hunter and one rabbit
        return self.agent_rep_size * 2

    def start_state(self):
        """Returns a random initial state. The state vector is a flat array of:
           concat(hunter positions, rabbit positions). Do not allow for overlapping hunters and rabbits."""

        # 1 for active status
        possible_poses = [(1, i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        rabbit_poses = random.choices(possible_poses, k=self.num_rabbits)

        possible_poses = [pos for pos in possible_poses if pos not in rabbit_poses]

        hunter_poses = random.choices(possible_poses, k=self.num_hunters)

        return np.array(hunter_poses + rabbit_poses).reshape(-1)

    def perform_action(self, state, a_indices):
        """Performs an action given by a_indices in state s. Returns:
           (s_next, reward)"""
        a = action_indices_to_coordinates(a_indices)

        # Get positions after hunter and rabbit actions
        hunter_pos = np.zeros(self.num_active_hunters * 3, dtype=np.int)
        for hunter in range(0, self.num_active_hunters):
            hunter_idx = hunter * 3
            if state[hunter_idx] == 0:
                hunter_pos[hunter_idx:hunter_idx + 3] = [0, -1, -1]
            else:
                hunter_pos[hunter_idx] = 1
                hunter_act = a[hunter_idx - hunter:hunter_idx - hunter + 2]
                sa = state[hunter_idx + 1:hunter_idx + 3] + hunter_act
                clipped = np.clip(sa, 0, self.grid_size - 1)
                hunter_pos[hunter_idx + 1:hunter_idx + 3] = clipped

        # Remove rabbits (and optionally hunters) that overlap
        reward = self.timestep_reward
        rabbit_pos = np.array(state[self.num_active_hunters * 3:])

        captured_rabbit_idxes = []
        inactive_hunter_idxes = []
        for i in range(0, len(hunter_pos), 3):
            hunter = hunter_pos[i:i + 3]
            for j in range(0, len(rabbit_pos), 3):
                rabbit = rabbit_pos[j:j + 3]
                if hunter[0] == 1 and rabbit[0] == 1 and array_equal(hunter, rabbit):
                    # A rabbit has been captured
                    # Remove captured rabbit and respective hunter
                    rabbit_pos[j:j + 3] = [0, -1, -1]
                    captured_rabbit_idxes += [j, j + 1, j + 2]
                    reward += self.capture_reward
                    hunter_pos[i:i + 3] = [0, -1, -1]
                    inactive_hunter_idxes += [i, i + 1, i + 2]

        rabbit_pos = np.delete(rabbit_pos, captured_rabbit_idxes, axis=0)
        hunter_pos = np.delete(hunter_pos, inactive_hunter_idxes, axis=0)
        self.num_active_hunters -= int(len(inactive_hunter_idxes) / 3)
        self.num_active_rabbits -= int(len(captured_rabbit_idxes) / 3)

        s_next = np.concatenate((hunter_pos, rabbit_pos))

        return s_next, reward

    def filter_actions(self, state, agent_no):
        """Filter the actions available for an agent in a given state. Returns a
           bitmap of available actions. Hunter should be active.
           E.g. an agent in a corner is not allowed to move into a wall."""
        avail_a = np.ones(9, dtype=int)
        hunter_pos = state[3 * agent_no + 1:3 * agent_no + 3]

        for i in range(len(RabbitHunter.action_space)):
            # Check if action moves us off the grid
            a = RabbitHunter.action_space[i]
            sa = hunter_pos + a
            if (sa[0] < 0 or sa[0] >= self.grid_size) or (sa[1] < 0 or sa[1] >= self.grid_size):
                avail_a[i] = 0
        return avail_a

    def is_end(self, state):
        """Given a state, return if the game should end."""
        if len(state) == 0:
            return True
        if self.end_when_capture is not None:
            num_rabbits_remaining = self._get_num_rabbits_from_state_size(len(state))
            if (self.num_rabbits - num_rabbits_remaining) >= self.end_when_capture:
                return True
        return False

    def _get_num_hunters_from_state_size(self, state_size):
        return int(state_size / self.agent_rep_size / 2)

    def _get_num_rabbits_from_state_size(self, state_size):
        return int(state_size / self.agent_rep_size / 2)

    def _get_hunters_state_from_state(self, state):
        return state[:self._get_num_hunters_from_state_size(len(state)) * self.agent_rep_size]

    def _get_rabbits_state_from_state(self, state):
        return state[self._get_num_hunters_from_state_size(len(state)) * self.agent_rep_size:]

    def _get_hunters_from_state(self, state):
        hunters_state = self._get_hunters_state_from_state(state)
        return np.split(hunters_state, self._get_num_hunters_from_state_size(len(state)))

    def _get_rabbits_from_state(self, state):
        rabbits_state = self._get_rabbits_state_from_state(state)
        return np.split(rabbits_state, self._get_num_rabbits_from_state_size(len(state)))

    def _get_poses_from_one_d_array(self, array):
        positions = []
        for idx in range(0, len(array), 3):
            # +1 to skip the status number
            positions.append(array[idx+1: idx+3].tolist())
        return positions

    def render(self, state, outfile=sys.stdout):
        hunter_poses = self._get_poses_from_one_d_array(state[:self.num_active_hunters * self.agent_rep_size])
        rabbit_poses = self._get_poses_from_one_d_array(state[self.num_active_hunters * self.agent_rep_size:])

        outfile.write(f'Rendering state: {state}')

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


def action_indices_to_coordinates(a_indices):
    """Converts a list of action indices to action coordinates."""
    coords = [RabbitHunter.action_space[i] for i in a_indices]
    return np.concatenate(coords)

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

