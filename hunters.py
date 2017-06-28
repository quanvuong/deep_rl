import numpy as np
import sys
import random


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

    action_space = [
        np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
        np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
        np.array([1, -1]), np.array([1, 0]), np.array([1, 1])
    ]

    def __init__(self, options):
        self.initial_options = options
        self.set_options(options)
        self.agent_rep_size = 3
        print(options.__dict__)

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
        return 6

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
        # print(a)
        # sys.stdout.flush()

        # Get positions after hunter and rabbit actions
        # a = np.concatenate((a, rabbit_a))
        hunter_pos = np.zeros(self.num_active_hunters * 3, dtype=np.int)
        for hunter in range(0, self.num_active_hunters):
            hunter_idx = hunter * 3
            if state[hunter_idx] == 0:
                hunter_pos[hunter_idx:hunter_idx + 3] = [0, -1, -1]
            else:
                hunter_pos[hunter_idx] = 1
                hunter_act = a[hunter_idx - hunter:hunter_idx - hunter + 2]
                # print(hunter_act)
                sa = state[hunter_idx + 1:hunter_idx + 3] + hunter_act
                # print(sa)
                clipped = np.clip(sa, 0, self.grid_size - 1)
                # print(clipped)
                # print(hunter_pos[hunter_idx + 1:hunter_idx + 3])
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

        # Return (s_next, reward)
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
            num_rabbits_remaining = self.get_num_rabbits_from_state_size(len(state))
            if (self.num_rabbits - num_rabbits_remaining) >= self.end_when_capture:
                return True
        return False

    def get_num_hunters_from_state_size(self, state_size):
        return int(state_size / self.agent_rep_size / 2)

    def get_num_rabbits_from_state_size(self, state_size):
        return int(state_size / self.agent_rep_size / 2)

    def get_hunters_state_from_state(self, state):
        return state[:self.num_active_hunters * self.agent_rep_size]

    def get_rabbits_state_from_state(self, state):
        return state[self.num_active_hunters * self.agent_rep_size:]

    def get_hunters_from_state(self, state):
        hunters_state = self.get_hunters_state_from_state(state)
        return np.split(hunters_state, self.num_active_hunters)

    def get_rabbits_from_state(self, state):
        rabbits_state = self.get_rabbits_state_from_state(state)
        return np.split(rabbits_state, self.num_active_rabbits)

    def _get_poses_from_one_d_array(self, array):
        positions = []
        for idx in range(0, len(array), 3):
            # +1 to skip the status
            positions.append(array[idx+1: idx+3].tolist())
        return positions

    def render(self, state, outfile=sys.stdout):
        hunter_poses = self._get_poses_from_one_d_array(state[:self.num_active_hunters * self.agent_rep_size])
        rabbit_poses = self._get_poses_from_one_d_array(state[self.num_active_hunters * self.agent_rep_size:])

        outfile.write('NEW STATE\n')

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

