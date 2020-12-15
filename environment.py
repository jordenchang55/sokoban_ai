import matplotlib.pyplot as plt
import numpy as np


class Environment:
	UP = np.array([0, 1])
	DOWN = np.array([0, -1])
	LEFT = np.array([-1, 0])
	RIGHT = np.array([1, 0])
	DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

	def __init__(self, xlim, ylim):
		self.fig = plt.figure()
		self.xlim = xlim
		self.ylim = ylim

	def is_goal_state(self, state):
		raise NotImplementedError

	def draw(self, state):
		raise NotImplementedError

	def next_state(self, state, action):
		raise NotImplementedError

	def get_player(self, state):
		raise NotImplementedError

	def get_neighbors(self, state):
		raise NotImplementedError

	def direction_to_str(self, direction):
		if all(direction == Environment.UP):
			return "UP"
		elif all(direction == Environment.DOWN):
			return "DOWN"
		elif all(direction == Environment.LEFT):
			return "LEFT"
		return "RIGHT"

	def rotate_direction_90(self, direction):
		if all(direction == Environment.UP):
			return Environment.RIGHT
		elif all(direction == Environment.RIGHT):
			return Environment.DOWN
		elif all(direction == Environment.DOWN):
			return Environment.LEFT
		return Environment.UP
