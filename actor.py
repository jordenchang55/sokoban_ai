from enum import Enum
import random
import numpy as np
from environment import MapType

from collections import namedtuple



class Actor():
	UP = np.array([0,1])
	DOWN = np.array([0, -1])
	LEFT = np.array([-1, 0])
	RIGHT = np.array([1, 0])

	def __init__(self, *args, **kwargs):
		#0, 1, 2, 3
		self.actions = [self.UP, self.RIGHT, self.DOWN, self.LEFT]


		#print(self.directions)

	def get_neighbors(self, player, boxes, sokoban_map):
		pass


	def get_action(self, state, sokoban_map, storage):

		return random.choice(self.actions)







class SimpleActor(Actor):

	def __init__(self, *args, **kwargs):
		pass

	def heuristic(self, state, sokoban_map, storage):
		pass


	def get_action(self, state, sokoban_map, storage):


		pass	




class QLearning(Actor):


	def __init__(self, *args, **kwargs):

		self.statemap = {}

		self.learning_rate = kwargs['learning_rate']
		self.discount_factor = kwargs['discount_factor']

	def next_state(self, state, action):
		pass
	def reward(self, state, action):
		pass

	def update(self, state, action):

		if str(state) not in self.statemap:
			self.statemap[str(state, action)] = 0


		qmax = np.amax(np.array([self.statemap[str((state, action))] for action in self.actions]))
		self.statemap[str(state, action)] += self.learning_rate*(reward(state, action) + self.discount_factor*(qmax) - self.statemap[str(state, action)])


	def get_action(self, state, sokoban_map, storage):


		action = random.choice(self.actions)

