import csv
import argparse
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from pathlib import Path
import numpy as np
#from enum import Enum

from actor import Actor
from environment import Environment
from environment import MapType
# class MapType(Enum):
# 	EMPTY = 0
# 	PLAYER = 1
# 	BOX = 2
# 	WALL = 3

# def goal(sokoban_map, storage):
# 	for place in storage:
# 		if sokoban_map[place] != MapType.BOX.value:
# 			return False

# 	return True

# def draw(fig, xlim, ylim, sokoban_map, storage):
	
# 	ax = plt.gca()
# 	ax.clear()
# 	#create square boundary
# 	lim = max(xlim, ylim)
# 	plt.xlim(0, lim+1)
# 	plt.ylim(0, lim+1)
# 	ax.set_xticks(np.arange(0, lim+1))
# 	ax.set_yticks(np.arange(0, lim+1))
# 	plt.grid(alpha=0.2)


# 	for i in range(sokoban_map.shape[0]):
# 		for j in range(sokoban_map.shape[1]):
# 			#print((i,j))
# 			if sokoban_map[i,j] == MapType.WALL.value:
# 				rect = patches.Rectangle((i+0.5, j+0.5),-1,-1,linewidth=0.5,edgecolor='slategray',facecolor='slategray')
# 				ax.add_patch(rect)
# 			elif sokoban_map[i,j] == MapType.BOX.value:
# 				rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='tan', facecolor='tan')
# 				ax.add_patch(rect)
# 			elif sokoban_map[i,j] == MapType.PLAYER.value:
# 				plt.plot(i, j, 'o', color='orange')

# 	for place in storage:
# 		circle = patches.Circle(place, 0.05, edgecolor='limegreen', facecolor='limegreen')
# 		ax.add_patch(circle)



	
# 	#plt.draw()
# 	#plt.show()
# 	if args.save_figure:
# 		plt.savefig('sokoban.png')
# 	else:
# 		plt.show(block=False)
# 		# background = fig.canvas.copy_from_bbox(ax.bbox)
# 		# fig.canvas.restore_region(background)
# 		# fig.canvas.draw()

# 		plt.pause(0.2)





# def make_move(sokoban_ai, sokoban_map, player):


# 	move = sokoban_ai.get_move(player, sokoban_map)
# 	#print(move)
# 	#print(move)
# 	next_position = move + player
# 	#print(next_position)
# 	if sokoban_map[tuple(next_position)] == MapType.BOX.value:
# 		#print("BOX")

# 		box_next_position = next_position + move

# 		if sokoban_map[tuple(box_next_position)] == MapType.EMPTY.value:
# 			sokoban_map[tuple(player)] = MapType.EMPTY.value
# 			sokoban_map[tuple(next_position)] = MapType.PLAYER.value
# 			sokoban_map[tuple(box_next_position)] = MapType.BOX.value
# 			player = next_position

# 			return next_position
# 		else:
# 			#impossible to move box
# 			return player

# 	elif sokoban_map[tuple(next_position)] == MapType.WALL.value:
# 		#print(tuple(next_position))
# 		#print("WALL")
# 		return player
# 	elif sokoban_map[tuple(next_position)] == MapType.EMPTY.value:
# 		#print("EMPTY")
# 		sokoban_map[tuple(player)] = MapType.EMPTY.value
# 		sokoban_map[tuple(next_position)] = MapType.PLAYER.value
# 		player = next_position
# 		return next_position
# 	return player

def main():

	#fig = plt.figure()

	filepath = Path(args.filename)
	#print(args.filename)
	if not filepath.exists():
		raise ValueError("Path does not exist.")
	if not filepath.is_file():
		raise ValueError("Path is not a valid file.")



	with open(filepath, 'r') as file:
		csv_input = csv.reader(file, delimiter=' ')

		for index, row in enumerate(csv_input):

			def unpack(points):

				return [tuple([int(points[index+1]), int(points[index])]) for index in range(1, len(points), 2)]

			#print(index, row)

			if index == 0:
				#sizeH, sizeV

				xlim = int(row[0])
				ylim = int(row[1])
				sokoban_map = np.zeros((xlim+1, ylim+1))
			if index == 1:
				#print(MapType.WALL.value)
				walls = unpack(row)
			elif index == 2:	
				boxes = unpack(row)
				for box in boxes:
					sokoban_map[box] = MapType.BOX.value
			elif index == 3:
				storage = unpack(row)
			elif index == 4:
				player = np.array([int(row[0]), int(row[1])])
				sokoban_map[(int(row[0]), int(row[1]))] = MapType.PLAYER.value

	environment = Environment(actor = Actor(), walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)

	num_iterations = 0
	while not environment.goal():
		environment.step()
		environment.draw()
		num_iterations += 1

	print("Goal reached.")
	print(f"iterations:{num_iterations:3d}")

	plt.show(block=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
	parser.add_argument('filename')
	parser.add_argument('--save_figure', '-s', action='store_true')
	args = parser.parse_args()
	main()