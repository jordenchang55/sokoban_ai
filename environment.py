import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
from collections import namedtuple
import copy
#from actor import State


State = namedtuple('State', ['player', 'boxes'])

class MapType(Enum):
	EMPTY = 0
	PLAYER = 1
	BOX = 2
	WALL = 3


UP = np.array([0,1])
DOWN = np.array([0, -1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

class Environment():


	def __init__(self, actor, walls, boxes, player, storage, xlim, ylim):
		self.actor = actor

		self.fig = plt.figure()
		self.map = np.zeros((xlim+1, ylim+1))
		self.xlim = xlim
		self.ylim = ylim

		self.state = State(player=player, boxes = np.array(boxes))

		self.storage = set(storage)


		for wall in walls:
			self.map[wall] = MapType.WALL.value
		for box in boxes:
			self.map[box] = MapType.BOX.value
		self.map[tuple(player)] = MapType.PLAYER.value
		self.player = player
		self.boxes = boxes

		self.original_map = copy.deepcopy(self.map)
		self.original_player = copy.deepcopy(self.player)
		self.original_boxes = copy.deepcopy(self.boxes)




	def reset(self):
		# print("reset!")
		# print(f"player:{self.player}")
		# print(f"reset_player:{self.original_player}")
		self.map = copy.deepcopy(self.original_map)
		self.player = copy.deepcopy(self.original_player)
		self.boxes = copy.deepcopy(self.original_boxes)

	def goal(self):
		for place in self.storage:
			if self.map[place] != MapType.BOX.value:
				return False

		return True




	def get_neighbors(self, location):
		return [location + direction for direction in DIRECTIONS]


	def is_frozen(self, location, previous=set([])):
		neighbors = self.get_neighbors(location)
		previous.add(tuple(location))
		if tuple(location) not in self.storage:
			for i in range(len(neighbors)):
				neighbor = tuple(neighbors[i])
				next_neighbor = tuple(neighbors[(i+1)%len(neighbors)])

				if self.map[neighbor] == MapType.WALL.value and self.map[next_neighbor] == MapType.WALL.value:
					return True
				elif self.map[neighbor] == MapType.WALL.value and self.map[next_neighbor] == MapType.BOX.value:
					if next_neighbor in previous:
						#dependency cycle!
						return True

					if self.is_frozen(np.array(next_neighbor), previous):
						return True
				elif self.map[neighbor] == MapType.BOX.value and self.map[next_neighbor] == MapType.WALL.value:
					if neighbor in previous:
						#dependency cycle!
						return True

					if self.is_frozen(np.array(neighbor), previous):
						return True
				elif self.map[neighbor] == MapType.BOX.value and self.map[next_neighbor] == MapType.BOX.value:
					if neighbor not in previous:
						frozen_neighbor = is_frozen(np.array(neighbor), previous)
					else:
						frozen_neighbor = True
					if next_neighbor not in previous:
						frozen_next_neighbor = is_frozen(np.array(neighbor), previous)
					else:
						frozen_next_neighbor = True

					if frozen_neighbor and frozen_next_neighbor:
						return True


	def check_deadlock(self):
		for box in self.boxes:
			if self.is_frozen(box):
				return True


			# if tuple(box) not in self.storage:
			# 	for i in range(len(neighbors)):
			# 		#print(neighbor)
			# 		if self.map[tuple(neighbors[i])] == MapType.WALL.value and \
			# 			self.map[tuple(neighbors[(i+1)%len(neighbors)])] == MapType.WALL.value:
			# 			return True
					#do not include boxes for now


		return False


	def step(self, evaluate=False):


		if not evaluate:
			action = self.actor.learn(State(self.player, self.boxes), self.map)
		else:
			action = self.actor.evaluate(State(self.player, self.boxes), self.map)
		#print(move)
		#print(move)
		next_position = action + self.player
		#print(next_position)
		if self.map[tuple(next_position)] == MapType.BOX.value:
			#print("BOX")

			box_next_position = next_position + action

			if self.map[tuple(box_next_position)] == MapType.EMPTY.value:
				self.map[tuple(self.player)] = MapType.EMPTY.value
				self.map[tuple(next_position)] = MapType.PLAYER.value
				self.map[tuple(box_next_position)] = MapType.BOX.value
				self.player = next_position

				for i in range(len(self.boxes)):
					if (self.boxes[i] == next_position).all():
						self.boxes[i] = box_next_position 

				return next_position
			else:
				#impossible to move box
				return self.player

		elif self.map[tuple(next_position)] == MapType.WALL.value:
			#print(tuple(next_position))
			#print("WALL")
			return self.player
		elif self.map[tuple(next_position)] == MapType.EMPTY.value:
			#print("EMPTY")
			self.map[tuple(self.player)] = MapType.EMPTY.value
			self.map[tuple(next_position)] = MapType.PLAYER.value
			self.player = next_position
			return next_position
		return self.player






	def draw(self, save_figure = False):
	
		ax = plt.gca()
		ax.clear()
		#create square boundary
		lim = max(self.xlim, self.ylim)
		plt.xlim(0, lim+1)
		plt.ylim(0, lim+1)
		ax.set_xticks(np.arange(0, lim+1))
		ax.set_yticks(np.arange(0, lim+1))
		plt.grid(alpha=0.2)


		for i in range(self.map.shape[0]):
			for j in range(self.map.shape[1]):
				#print((i,j))
				if self.map[i,j] == MapType.WALL.value:
					rect = patches.Rectangle((i+0.5, j+0.5),-1,-1,linewidth=0.5,edgecolor='slategray',facecolor='slategray')
					ax.add_patch(rect)
				elif self.map[i,j] == MapType.BOX.value:
					rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='tan', facecolor='tan')
					ax.add_patch(rect)
				elif self.map[i,j] == MapType.PLAYER.value:
					plt.plot(i, j, 'o', color='orange')

		for place in self.storage:
			circle = patches.Circle(place, 0.05, edgecolor='limegreen', facecolor='limegreen')
			ax.add_patch(circle)



		
		#plt.draw()
		#plt.show()
		if save_figure:
			plt.savefig('sokoban.png')
		else:
			plt.show(block=False)
			# background = fig.canvas.copy_from_bbox(ax.bbox)
			# fig.canvas.restore_region(background)
			# fig.canvas.draw()

			plt.pause(0.05)