import unittest
import numpy as np
import csv
from environment import Environment
from deepenvironment import DeepEnvironment
# from environment import Environment.DOWN, Environment.RIGHT, Environment.LEFT, Environment.UP, DIRECTIONS

from deepqagent import DeepQAgent, PrioritizedReplayBuffer
import copy
# import logging

from sokoban import load


class DeepQAgentTest(unittest.TestCase):

    def setUp(self):
        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban01.txt')

        self.environment = DeepEnvironment(walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim,
                                           ylim=ylim)
        self.agent = DeepQAgent(environment=self.environment, discount_factor=0.95, verbose=False)

    def tearDown(self):
        self.agent = None

    def test_reward(self):

        deadlock_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT]
        goal_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.RIGHT,
                         Environment.DOWN, Environment.RIGHT, Environment.DOWN, Environment.DOWN, Environment.LEFT,
                         Environment.LEFT, Environment.LEFT, Environment.UP, Environment.LEFT, Environment.DOWN,
                         Environment.RIGHT, Environment.UP, Environment.UP, Environment.UP, Environment.UP,
                         Environment.LEFT, Environment.UP, Environment.RIGHT]

        state = copy.deepcopy(self.environment.state)

        for direction in Environment.DIRECTIONS:
            self.assertAlmostEquals(self.agent.reward(state, direction), -0.01,
                                    msg="Reward for standard movement is incorrect.")

        for index, action in goal_sequence:
            if index == 6 or index == 14:
                self.assertAlmostEquals(self.agent.reward(state, action), 1,
                                        msg="Reward for standard movement is incorrect.")
            elif index == len(goal_sequence) - 1:
                self.assertAlmostEquals(self.agent.reward(state, action), 1.,
                                        msg="Reward for standard movement is incorrect.")
            else:
                self.assertAlmostEquals(self.agent.reward(state, action), -0.01,
                                        msg="Reward for standard movement is incorrect.")
            state = self.environment.next_state(state, action)

        # state = copy.deepcopy(self.environment.state)

    # def test_target(self):
    #     goal_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.RIGHT, Environment.DOWN, Environment.RIGHT, Environment.DOWN, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT, Environment.UP, Environment.LEFT, Environment.DOWN, Environment.RIGHT, Environment.UP, Environment.UP, Environment.UP, Environment.UP, Environment.LEFT, Environment.UP, Environment.RIGHT]


class PrioritizedReplayBufferTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.max_size = 32
        self.buffer = PrioritizedReplayBuffer(self.max_size)

    def test_sample_values_unique(self):
        for i in range(self.max_size):
            self.buffer.add(i * 2)
        values, indices = self.buffer.sample(10)
        self.assertEqual(len(np.unique(values)), len(values))
        for v in values:
            self.assertIn(v, self.buffer.buffer)

    def test_sample_more_than_buffer_size(self):
        self.buffer.add(5)
        self.buffer.add(10)

        values, indices = self.buffer.sample(10)
        self.assertEqual(len(values), 2)
        self.assertEqual(len(indices), 2)

    def test_sample_fewer_than_max_size(self):
        for i in range(10):
            self.buffer.add(i * 2)
        values, indices = self.buffer.sample(10)
        self.assertEqual(len(np.unique(values)), len(values))
        for v in values:
            self.assertIn(v, self.buffer.buffer)

    def test_add_more_than_max_size(self):
        for i in range(40):
            self.buffer.add(i * 2)
        values, indices = self.buffer.sample(self.max_size + 2)

        self.assertEqual(len(values), self.max_size)
        self.assertEqual(len(indices), self.max_size)
        for v in values:
            self.assertIn(v, self.buffer.buffer)

    def test_prioritized_sample(self):
        for i in range(self.max_size):
            self.buffer.add(i * 2)
        self.buffer.update_priorities(errors=[10, 20, 30], indices=[0, 1, 5])
        values, _ = self.buffer.sample(10)

        self.assertIn(0, values)
        self.assertIn(2, values)
        self.assertIn(10, values)
