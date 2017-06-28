import unittest
from kenny_drl.hunters import RabbitHunter, GameOptions


class HunterTest(unittest.TestCase):

    def test_initial_start_no_overlapping_hunter_rabbit(self):

        game_options = GameOptions(
            num_hunters=6,
            num_rabbits=6,
            grid_size=8,
            timestep_reward=0,
            capture_reward=1,
            end_when_capture=None
        )
        game = RabbitHunter(game_options)

        # Naive test of 10 random initial start_state
        for _ in range(10):
            start_state = game.start_state()
            hunters = game.get_hunters_from_state(start_state)
            rabbits = game.get_rabbits_from_state(start_state)

            for hunter in hunters:
                for rabbit in rabbits:
                    self.assertFalse((hunter == rabbit).all())

    def test_timestep_penalty(self):

        num_hunters = 3
        num_rabbits = 3
        grid_size = 4

        state = [1, 0, 0, 1, 1, 1, 1, 2, 3, 1, 0, 1, 1, 1, 2, 1, 3, 3]

        assert(len(state) == (num_rabbits + num_hunters) * 3)

        # Test capture reward + no timestep penalty
        game_options = GameOptions(
            num_hunters=num_hunters,
            num_rabbits=num_rabbits,
            grid_size=grid_size,
            timestep_reward=0,
            capture_reward=1,
            end_when_capture=None
        )

        game = RabbitHunter(game_options)
        act_idxes = [5, 5, 7]

        ns, reward = game.perform_action(state, act_idxes)

        self.assertEqual(game.capture_reward, 1)
        self.assertEqual(game.timestep_reward, 0)
        self.assertEqual(reward, 3)

        # Test no capture reward + timestep penalty
        game_options = GameOptions(
            num_hunters=num_hunters,
            num_rabbits=num_rabbits,
            grid_size=grid_size,
            timestep_reward=-1,
            capture_reward=0,
            end_when_capture=None
        )

        game = RabbitHunter(game_options)
        act_idxes = [4, 4, 4]

        ns, reward = game.perform_action(state, act_idxes)
        self.assertEqual(game.capture_reward, 0)
        self.assertEqual(game.timestep_reward, -1)
        self.assertEqual(reward, -1)

        # Test capture reward + timestep penalty
        game_options = GameOptions(
            num_hunters=num_hunters,
            num_rabbits=num_rabbits,
            grid_size=grid_size,
            timestep_reward=-1,
            capture_reward=1,
            end_when_capture=None
        )

        game = RabbitHunter(game_options)
        act_idxes = [5, 5, 7]

        ns, reward = game.perform_action(state, act_idxes)
        self.assertEqual(game.capture_reward, 1)
        self.assertEqual(game.timestep_reward, -1)
        self.assertEqual(reward, 2)


if __name__ == '__main__':
    unittest.main()
