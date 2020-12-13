



import unittest
import numpy as np
import csv
from environment import Environment
from deepenvironment import DeepEnvironment
#from environment import Environment.DOWN, Environment.RIGHT, Environment.LEFT, Environment.UP, DIRECTIONS

from deepqagent import DeepQAgent
import copy
#import logging

from sokoban import load




class DeepQAgentTest(unittest.TestCase):




    def setUp(self):
        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban01.txt')

        self.environment = DeepEnvironment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)
        self.agent = DeepQAgent(environment = self.environment, discount_factor=0.95, verbose=False)


    def tearDown(self):
        self.agent = None



    def test_reward(self):
        
        deadlock_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT]
        goal_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.RIGHT, Environment.DOWN, Environment.RIGHT, Environment.DOWN, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT, Environment.UP, Environment.LEFT, Environment.DOWN, Environment.RIGHT, Environment.UP, Environment.UP, Environment.UP, Environment.UP, Environment.LEFT, Environment.UP, Environment.RIGHT]


        state = copy.deepcopy(self.environment.state)
        
        for direction in Environment.DIRECTIONS:
            self.assertAlmostEquals(self.agent.reward(state, direction), -0.01, msg="Reward for standard movement is incorrect.")

        for index, action in goal_sequence:
            if index == 6 or index == 14:
                self.assertAlmostEquals(self.agent.reward(state, action), 1, msg="Reward for standard movement is incorrect.")
            elif index == len(goal_sequence) - 1:
                self.assertAlmostEquals(self.agent.reward(state, action), 1., msg="Reward for standard movement is incorrect.")
            else:
                self.assertAlmostEquals(self.agent.reward(state, action), -0.01, msg="Reward for standard movement is incorrect.")
            state = self.environment.next_state(state, action)

        #state = copy.deepcopy(self.environment.state)

    # def test_target(self):
    #     goal_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.RIGHT, Environment.DOWN, Environment.RIGHT, Environment.DOWN, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT, Environment.UP, Environment.LEFT, Environment.DOWN, Environment.RIGHT, Environment.UP, Environment.UP, Environment.UP, Environment.UP, Environment.LEFT, Environment.UP, Environment.RIGHT]

        


