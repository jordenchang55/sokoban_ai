import csv
import argparse
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from pathlib import Path
import numpy as np
#from enum import Enum

import actor
from actor import QActor
from environment import Environment
from environment import MapType

# def debug_print(fmt_string):
# 	print(fmt_string)
# control variables
iteration_max = 1000 #deadlock by iteration 

def main():

	max_episodes = abs(args.episodes)

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
			if index == 1:
				#print(MapType.WALL.value)
				walls = unpack(row)
			elif index == 2:	
				boxes = unpack(row)
			elif index == 3:
				storage = unpack(row)
			elif index == 4:
				player = np.array([int(row[0]), int(row[1])])

	environment = Environment(actor = QActor(storage = storage, learning_rate = 1., discount_factor=0.9), walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)

	episode_bookmarks = []
	episode_iterations = []

	num_episodes = 0
	num_iterations = 0
	goals_reached = 0
	while num_episodes < max_episodes:
		num_iterations = 0
		while not (environment.goal() or environment.check_deadlock()):
			environment.step()
			if args.draw:
				environment.draw()
			num_iterations += 1

			if num_iterations > iteration_max:
				environment.draw()
				break




		if environment.goal():
			goals_reached += 1
			episode_bookmarks.append(num_episodes)
			episode_iterations.append(num_iterations)

			print(f"{num_episodes:4d}:goal reached")
		else:
			if num_episodes%100 == 0:
				print(f"{num_episodes:4d}:deadlock reached")



		environment.reset()
		num_episodes += 1



		if num_episodes > 0 and num_episodes % 500 == 0:
			#evaluate!
			previous = []
			if len(previous) > 3:
				previous.pop()
			while not (environment.goal() or environment.check_deadlock()):
				move = environment.step(evaluate=True)
				environment.draw()
				if num_iterations > 200 or any([(move == prev).all() for prev in previous]):
					break
				previous.append(move)
			print("Evaluation results:")
			if environment.goal():
				print("Goal reached.")
			print(f"num_iterations:{num_iterations}")
	episode_iterations = np.array(episode_iterations)


	print("-"*30)
	print("Simulation ended.")
	print(f"episodes           :{num_episodes}")
	print(f"goals              :{goals_reached}")
	print(f"average iterations :{np.mean(episode_iterations):3f}")

	plt.show(block=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
	parser.add_argument('filename')
	parser.add_argument('--episodes', '-e', action='store', type=int, default=5000)
	parser.add_argument('--save_figure', '-s', action='store_true')
	parser.add_argument('--draw', '-d', action='store_true')
	args = parser.parse_args()
	main()