



import unittest
import numpy as np
import csv
from environment import Environment
from environment import DOWN, RIGHT, LEFT, UP, DIRECTIONS

from deepqagent import DeepQAgent
import copy
#import logging

from sokoban import load




class DeepQAgentTest(unittest.TestCase):




    def setUp(self):
        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban01.txt')

        self.environment = Environment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)
        self.agent = DeepQAgent(environment = self.environment, discount_factor=0.95, verbose=False)


    def tearDown(self):
        self.agent = None



    def test_reward(self):
        
        deadlock_sequence = [LEFT, DOWN, LEFT, LEFT, LEFT]
        goal_sequence = [LEFT, DOWN, LEFT, LEFT, RIGHT, DOWN, RIGHT, DOWN, DOWN, LEFT, LEFT, LEFT, UP, LEFT, DOWN, RIGHT, UP, UP, UP, UP, LEFT, UP, RIGHT]


        state = copy.deepcopy(self.environment.state)
        
        for direction in DIRECTIONS:
            self.assertEquals(self.agent.reward(state, direction), -1., msg="Reward for standard movement is incorrect, should be -1.")

        for index, action in goal_sequence:
            if index == 6 or index == 14:
                self.assertEquals(self.agent.reward(state, action), 50., msg="Reward for standard movement is incorrect, should be 50.")
            elif index == len(goal_sequence) - 1:
                self.assertEquals(self.agent.reward(state, action), 500., msg="Reward for standard movement is incorrect, should be 500.")
            else:
                self.assertEquals(self.agent.reward(state, action), -1., msg="Reward for standard movement is incorrect, should be -1.")
            state = self.environment.next_state(state, action)

        #state = copy.deepcopy(self.environment.state)

