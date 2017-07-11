import unittest
from kenny_drl.hunters import RabbitHunter, GameOptions


class HunterTest(unittest.TestCase):

    def setUp(self):
        game_options = GameOptions(
            num_hunters=6,
            num_rabbits=6,
            grid_size=8,
            timestep_reward=0,
            capture_reward=1,
            end_when_capture=None
        )
        self.game = RabbitHunter(game_options)

    def test_initial_start_no_overlapping_hunter_rabbit(self):

        # Naive test of 10 random initial start_state
        for _ in range(10):
            start_state = self.game.start_state()
            hunters = RabbitHunter.get_hunters_from_state(start_state)
            rabbits = RabbitHunter.get_rabbits_from_state(start_state)

            for hunter in hunters:
                for rabbit in rabbits:
                    self.assertFalse(hunter == rabbit)

if __name__ == '__main__':
    unittest.main()
