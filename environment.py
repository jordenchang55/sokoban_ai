import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
from collections import namedtuple
State = namedtuple('State', ['player', 'boxes'])

class MapType(Enum):
	EMPTY = 0
	PLAYER = 1
	BOX = 2
	WALL = 3


class Environment():


	def __init__(self, actor, walls, boxes, player, storage, xlim, ylim, save_figure=False):
		self.actor = actor

		self.fig = plt.figure()
		self.map = np.zeros((xlim+1, ylim+1))
		self.xlim = xlim
		self.ylim = ylim

		self.state = State(player=player, boxes = np.array(boxes))

		self.storage = storage

		self.save_figure = save_figure

		for wall in walls:
			self.map[wall] = MapType.WALL.value
		for box in boxes:
			self.map[box] = MapType.BOX.value
		self.map[tuple(player)] = MapType.PLAYER.value
		self.player = player
		self.boxes = boxes

	def goal(self):
		for place in self.storage:
			if self.map[place] != MapType.BOX.value:
				return False

		return True

	def step(self):


		action = self.actor.get_action(State(self.player, self.boxes), self.map, self.storage)
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


				#self.boxes == 

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






	def draw(self):
	
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
		if self.save_figure:
			plt.savefig('sokoban.png')
		else:
			plt.show(block=False)
			# background = fig.canvas.copy_from_bbox(ax.bbox)
			# fig.canvas.restore_region(background)
			# fig.canvas.draw()

			plt.pause(0.2)