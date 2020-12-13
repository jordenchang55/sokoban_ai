

import unittest
import numpy as np
from stateenvironment import StateEnvironment
from environment import Environment
from stateenvironment   import State   
import copy
import logging
from sokoban import load
class StateEnvironmentTest(unittest.TestCase):

    def setUp(self):

        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban01.txt')
        self.environment = StateEnvironment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)

        self.deadlock_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT]
        self.goal_sequence = [Environment.LEFT, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.RIGHT, Environment.DOWN, Environment.RIGHT, Environment.DOWN, Environment.DOWN, Environment.LEFT, Environment.LEFT, Environment.LEFT, Environment.UP, Environment.LEFT, Environment.DOWN, Environment.RIGHT, Environment.UP, Environment.UP, Environment.UP, Environment.UP, Environment.LEFT, Environment.UP, Environment.RIGHT]

    def test_init(self):

        self.assertEquals(self.environment.state.map[7, 7], State.PLAYER, msg=f"State does not reflect player's position.")
        self.assertEquals(self.environment.state.map.shape, (9, 9))


        player = self.environment.get_player(self.environment.state)
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

        next_state = self.environment.next_state(self.environment.state, Environment.UP)
        self.assertTrue((original_state == self.environment.state).all(), msg="given parameter has been altered.")
        self.assertTrue((original_state == next_state).all(), msg="next_state is not the same as original_state.")

        next_state = self.environment.next_state(self.environment.state, Environment.DOWN)

        self.assertTrue(next_state[0, 7, 6], msg="New position does not contain player.")
        self.assertFalse(next_state[0, 7, 7], msg="Old position is not empty.")
        self.assertFalse(next_state[1, 7, 7], msg="Old position is not empty.")

    def test_is_goal_state(self):


        self.assertFalse(self.environment.is_goal_state(self.environment.state), msg="is_goal_state() returns true, despite not being a valid goal state.")
        current_state = copy.deepcopy(self.environment.state)

        for index, action in enumerate(self.goal_sequence):
            current_state = self.environment.next_state(current_state, action)

            if index < len(self.goal_sequence) - 1:
                self.assertFalse(self.environment.is_goal_state(current_state), msg="is_goal_state() returns true, despite not being a valid goal state.")
            else:
                self.assertTrue(self.environment.is_goal_state(current_state), msg="is_goal_state() returns false, despite being a valid goal state.")

        self.assertTrue((self.environment.state[2,:,:]==current_state[2,:,:]).any(), msg="walls are not equal...")


    def test_is_deadlock(self):
        current_state = copy.deepcopy(self.environment.state)

        for index, action in enumerate(self.deadlock_sequence):
            current_state = self.environment.next_state(current_state, action)
            if index < len(self.deadlock_sequence) - 1:
                self.assertFalse(self.environment.is_deadlock(current_state), msg="Should not be in deadlock, but is_deadlock() returns true.")
            else:
                self.assertTrue(self.environment.is_deadlock(current_state), msg="Should be in deadlock, but is_deadlock() reutrns false.")

        walls, boxes, storage, player, xlim, ylim = load('inputs/sokoban34.txt')
        self.environment = Environment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)

        current_state = copy.deepcopy(self.environment.state)

        current_state = self.environment.next_state(current_state, Environment.RIGHT)
        self.environment.pause = 2
        self.environment.draw(current_state)
        import matplotlib.pyplot as plt 
       
        self.assertFalse(self.environment.is_deadlock(current_state), msg="Should not be in deadlock in this state.")
        current_state = self.environment.next_state(current_state, Environment.DOWN)
        current_state = self.environment.next_state(current_state, Environment.RIGHT)

        self.assertFalse(self.environment.is_deadlock(current_state), msg="Should not be in deadlock in this state.")

        self.environment.draw(current_state)

    def test_is_valid(self):
        for i in range(10):
            for j in range(10):
                if i <= self.environment.xlim and j <= self.environment.ylim:
                    self.assertTrue(self.environment.is_valid((i, j)), msg="Should be valid location.")
                else:
                    self.assertFalse(self.environment.is_valid((i, j)), msg="Should be invalid location.")

    def test_count_boxes_scored(self):
        current_state = copy.deepcopy(self.environment.state)

        for index, action in enumerate(self.goal_sequence):
            current_state = self.environment.next_state(current_state, action)

            if index == len(self.goal_sequence) - 1:
                self.assertEquals(self.environment.count_boxes_scored(current_state), 3, msg="is_goal_state() returns true, despite not being a valid goal state.")
            # else:
            #     self.assertTrue(self.environment.is_goal_state(current_state), msg="is_goal_state() returns false, despite being a valid goal state.")

    def tearDown(self):

        self.environment = None

