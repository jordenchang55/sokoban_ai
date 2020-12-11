import copy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# from actor import State


# class MapType(Enum):
# 	EMPTY = 0
# 	PLAYER = 1
# 	BOX = 2
# 	WALL = 3

EMPTY = 0
PLAYER = 1
BOX = 2
WALL = 3
GOAL = 4
BOX_IN_GOAL = 5

UP = np.array([0, 1])
DOWN = np.array([0, -1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


def direction_to_str(direction):
	if all(direction == UP):
		return "UP"
	elif all(direction == DOWN):
		return "DOWN"
	elif all(direction == LEFT):
		return "LEFT"
	return "RIGHT"


class Environment():

	def __init__(self, walls, boxes, player, storage, xlim, ylim):

		self.fig = plt.figure()
		self.map = np.zeros((xlim + 1, ylim + 1))
		self.xlim = xlim
		self.ylim = ylim

		self.storage = set(storage)

		for wall in walls:
			self.map[wall] = WALL
		for box in boxes:
			self.map[box] = BOX
		for s in storage:
			if s in boxes:
				self.map[s] = BOX_IN_GOAL
			else:
				self.map[s] = GOAL
		self.map[tuple(player)] = PLAYER
		# self.state[0] = player
		# self.state[1:] = boxes
		self.state = np.array([player, *boxes])
		self.state_hash = self.state.tobytes()  # inbetween variable for hashing

		self.box_in_goal = [False for box in boxes]

		self.deadlock_table = {}

		self.original_map = copy.deepcopy(self.map)
		self.original_state = copy.deepcopy(self.state)

	def reset(self):
		# logging.info("reset!")
		# logging.info(f"player:{self.state[0]}")
		# logging.info(f"reset_player:{self.original_player}")
		self.map = copy.deepcopy(self.original_map)
		self.state = copy.deepcopy(self.original_state)
		self.box_in_goal = [False for box in self.state[1:]]

	def is_goal(self):
		return len(self.storage) == (self.map == BOX_IN_GOAL).sum()


	def get_neighbors(self, location):
		return [location + direction for direction in DIRECTIONS]

	def is_frozen(self, location, previous=None):

		if location.tobytes() in self.deadlock_table[self.state_hash]:
			return self.deadlock_table[self.state_hash][location.tobytes()]

		# if not previous:
		# 	previous = set([])
		neighbors = self.get_neighbors(location)
		previous.add(tuple(location))
		if tuple(location) not in self.storage:
			for i in range(len(neighbors)):
				neighbor = tuple(neighbors[i])
				next_neighbor = tuple(neighbors[(i + 1) % len(neighbors)])

				if self.map[neighbor] == WALL and self.map[next_neighbor] == WALL:
					# logging.info("case 1")
					return True
				elif self.map[neighbor] == WALL and self.map[next_neighbor] == BOX:
					# logging.info("case 2")
					if next_neighbor in previous:
						# depndency cycle!
						self.deadlock_table[self.state_hash][location.tobytes()] = True
						return True
					if self.is_frozen(np.array(next_neighbor), previous):
						self.deadlock_table[self.state_hash][location.tobytes()] = True
						return True
				elif self.map[neighbor] == BOX and self.map[next_neighbor] == WALL:
					# logging.info("case 3")

					if neighbor in previous:
						# dependency cycle!
						self.deadlock_table[self.state_hash][location.tobytes()] = True
						return True

					if self.is_frozen(np.array(neighbor), previous):
						self.deadlock_table[self.state_hash][location.tobytes()] = True
						return True
				elif self.map[neighbor] == BOX and self.map[next_neighbor] == BOX:
					# logging.info("case 4")
					# logging.info(neighbor in previous)
					# logging.info(next_neighbor in previous)
					if neighbor in previous:
						frozen_neighbor = True
					else:
						frozen_neighbor = self.is_frozen(np.array(neighbor), previous)
					if next_neighbor in previous:
						frozen_next_neighbor = True
					else:
						frozen_next_neighbor = self.is_frozen(np.array(next_neighbor), previous)

					if frozen_neighbor and frozen_next_neighbor:
						self.deadlock_table[self.state_hash][location.tobytes()] = True
						return True

		previous.remove(tuple(location))
		self.deadlock_table[self.state_hash][location.tobytes()] = False

		return False

	def is_deadlock(self):
		# if not self.frozen_nodes:
		# 	self.frozen_nodes = set([])
		self.state_hash = self.state.tobytes()

		if self.state_hash not in self.deadlock_table:
			self.deadlock_table[self.state_hash] = {}
		for box in self.state[1:]:
			if box.tobytes() in self.deadlock_table[self.state_hash] and self.deadlock_table[self.state_hash][
				box.tobytes()]:
				return True
			elif self.is_frozen(box, previous=set([])):

				# self.frozen_nodes = None
				return True

		# self.frozen_nodes = None
		return False

	def step(self, action):
		next_position = self.state[0] + action
		if self.map[tuple(next_position)] == BOX:
			# logging.info("BOX")

			box_next_position = next_position + action

			if self.map[tuple(box_next_position)] == EMPTY or self.map[tuple(box_next_position)] == GOAL:
				self.map[tuple(self.state[0])] = EMPTY
				self.map[tuple(next_position)] = PLAYER
				if self.map[tuple(box_next_position)] == EMPTY:
					self.map[tuple(box_next_position)] = BOX
				else:
					self.map[tuple(box_next_position)] = BOX_IN_GOAL
				self.state[0] = next_position

				for i in range(len(self.state[1:])):
					if (self.state[i + 1] == next_position).all():
						self.state[i + 1] = box_next_position
					if tuple(box_next_position) in self.storage:
						self.box_in_goal[i] = True

				return next_position
			else:
				# impossible to move box
				return self.state[0]

		elif self.map[tuple(next_position)] == WALL:
			# logging.info(tuple(next_position))
			# logging.info("WALL")
			return self.state[0]
		elif self.map[tuple(next_position)] == EMPTY:
			# logging.info("EMPTY")
			self.map[tuple(self.state[0])] = EMPTY
			self.map[tuple(next_position)] = PLAYER
			self.state[0] = next_position
			return next_position
		return self.state[0]

	# def step(self, evaluate=False):

	# 	if not evaluate:
	# 		action = self.actor.learn(State(self.state[0], self.state[1:]), self.map)
	# 	else:
	# 		action = self.actor.evaluate(State(self.state[0], self.state[1:]), self.map)
	# 	#logging.info(move)
	# 	#logging.info(move)
	# 	next_position = action + self.state[0]
	# 	#logging.info(next_position)

	def draw(self, save_figure=False):

		ax = plt.gca()
		ax.clear()
		# create square boundary
		lim = max(self.xlim, self.ylim)
		plt.xlim(0, lim + 1)
		plt.ylim(0, lim + 1)
		ax.set_xticks(np.arange(0, lim + 1))
		ax.set_yticks(np.arange(0, lim + 1))
		plt.grid(alpha=0.2)

		for i in range(self.map.shape[0]):
			for j in range(self.map.shape[1]):
				# logging.info((i,j))
				if self.map[i, j] == WALL:
					rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='slategray',
											 facecolor='slategray')
					ax.add_patch(rect)
				elif self.map[i, j] == BOX:
					rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='tan',
											 facecolor='tan')
					ax.add_patch(rect)
				elif self.map[i, j] == PLAYER:
					plt.plot(i, j, 'o', color='orange')
				elif self.map[i, j] == BOX_IN_GOAL:
					rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='orange',
											 facecolor='tan')
					ax.add_patch(rect)
		for place in self.storage:
			circle = patches.Circle(place, 0.05, edgecolor='limegreen', facecolor='limegreen')
			ax.add_patch(circle)

		# plt.draw()
		# plt.show()
		if save_figure:
			plt.savefig('sokoban.png')
		else:
			plt.show(block=False)
			# background = fig.canvas.copy_from_bbox(ax.bbox)
			# fig.canvas.restore_region(background)
			# fig.canvas.draw()

			plt.pause(0.05)
