

import unittest
import numpy as np
from environment import Environment
from environment import DOWN
from environment import RIGHT
from environment import LEFT
from environment import UP
import copy
import logging
from sokoban import load
class EnvironmentTest(unittest.TestCase):

    def setUp(self):

        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban01.txt')
        self.environment = Environment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)

        self.deadlock_sequence = [LEFT, DOWN, LEFT, LEFT, LEFT]
        self.goal_sequence = [LEFT, DOWN, LEFT, LEFT, RIGHT, DOWN, RIGHT, DOWN, DOWN, LEFT, LEFT, LEFT, UP, LEFT, DOWN, RIGHT, UP, UP, UP, UP, LEFT, UP, RIGHT]

    def test_init(self):

        self.assertEquals(self.environment.state[7, 7, 0], 1, msg=f"State does not reflect player's position.")
        self.assertEquals(self.environment.state.shape, (9, 9, 2))


        player = np.unravel_index(np.argmax(self.environment.state[:,:, 0]), self.environment.state[:,:,0].shape)
        self.assertEquals(player, (7, 7), msg=f"{player} not the same as (7,7).")
        

        #print
        # print(self.environment.state)

    def test_get_neighbors(self):

        location = np.array([1, 1])
        output = self.environment.get_neighbors(location)

        self.assertEquals(len(output), 4, msg="Output is not an array of size 4.")
        for neighbor in output:
            self.assertEquals(neighbor.shape, location.shape, msg="Shape of neighbor vector is incorrect.")

        correct_output = [np.array([1, 2]), np.array([2, 1]), np.array([1, 0]), np.array([0, 1])]
        
        for index, neighbor in enumerate(output):

            self.assertTrue((neighbor == correct_output[index]).all(), msg=f"{neighbor} is not the same as {correct_output[index]}.")


    def test_next_state(self):


        original_state = copy.deepcopy(self.environment.state)

        next_state = self.environment.next_state(self.environment.state, UP)
        self.assertTrue((original_state == self.environment.state).all(), msg="given parameter has been altered.")
        self.assertTrue((original_state == next_state).all(), msg="next_state is not the same as original_state.")

        next_state = self.environment.next_state(self.environment.state, DOWN)

        self.assertTrue(next_state[7, 6, 0], msg="New position does not contain player.")
        self.assertFalse(next_state[7, 7, 0], msg="Old position is not empty.")
        self.assertFalse(next_state[7, 7, 1], msg="Old position is not empty.")

    def test_is_goal_state(self):


        self.assertFalse(self.environment.is_goal_state(self.environment.state), msg="is_goal_state() returns true, despite not being a valid goal state.")
        current_state = copy.deepcopy(self.environment.state)

        for index, action in enumerate(self.goal_sequence):
            current_state = self.environment.next_state(current_state, action)

            if index < len(self.goal_sequence) - 1:
                self.assertFalse(self.environment.is_goal_state(current_state), msg="is_goal_state() returns true, despite not being a valid goal state.")
            else:
                self.assertTrue(self.environment.is_goal_state(current_state), msg="is_goal_state() returns false, despite being a valid goal state.")


    def test_is_deadlock(self):
        current_state = copy.deepcopy(self.environment.state)

        for index, action in enumerate(self.deadlock_sequence):
            current_state = self.environment.next_state(current_state, action)
            if index < len(self.goal_sequence) - 1:
                self.assertFalse(self.environment.is_goal_state(current_state), msg="Should not be in deadlock, but is_deadlock() returns true.")
            else:
                self.assertTrue(self.environment.is_goal_state(current_state), msg="Should be in deadlock, but is_deadlock() reutrns false.")


    def tearDown(self):

        self.environment = None

