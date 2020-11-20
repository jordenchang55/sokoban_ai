import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np


def draw(xlim, ylim, walls, boxes, storage, player):
	ax = plt.gca()

	#create square boundary
	lim = max(xlim, ylim)
	plt.xlim(-1, lim+1)
	plt.ylim(-1, lim+1)
	ax.set_xticks(np.arange(-1, lim+1))
	ax.set_yticks(np.arange(-1, lim+1))
	plt.grid(alpha=0.2)


	for wall in walls:
		rect = patches.Rectangle(wall,-1,-1,linewidth=1,edgecolor='slategray',facecolor='slategray')
		ax.add_patch(rect)
	
	for box in boxes:
		rect = patches.Rectangle(box, -1, -1, linewidth=1, edgecolor='tan', facecolor='tan')
		ax.add_patch(rect)

	for place in storage:
		circle = patches.Circle((place[0]-0.5, place[1]-0.5),0.1, edgecolor='limegreen', facecolor='limegreen')
		ax.add_patch(circle)
	# x, y = zip(*boxes)
	# plt.scatter(x, y)

	# x, y = zip(*storage)
	# plt.scatter(x, y)

	# x, y = zip(*player)
	# plt.scatter(x, y)
	plt.plot(player[0] - 0.5, player[1] - 0.5, 'o', color='orange')

	


	
	plt.draw()
	#plt.show()
	if args.save_figure:
		plt.savefig('sokoban.png')
	else:
		plt.show()
def main():
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

				return [[int(points[index+1]), int(points[index])] for index in range(1, len(points), 2)]

			print(index, row)

			if index == 0:
				#sizeH, sizeV

				xlim = int(row[0])
				ylim = int(row[1])

			if index == 1:

				walls = unpack(row)
			elif index == 2:	
				boxes = unpack(row)
			elif index == 3:
				storage = unpack(row)
			elif index == 4:
				player = [int(row[0]), int(row[1])]



	draw(xlim, ylim, walls, boxes, storage, player)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
	parser.add_argument('filename')
	parser.add_argument('--save_figure', '-s', action='store_true')
	args = parser.parse_args()
	main()