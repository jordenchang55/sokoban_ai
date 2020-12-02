import csv
import argparse
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from pathlib import Path
import numpy as np
#from enum import Enum

import agent
from agent import QAgent
from environment import Environment
from environment import MapType
import time

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

	environment = Environment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)
	# agent = QAgent(environment = environment, learning_rate = 1., discount_factor=0.9, replay_rate = 0.2)

	def converge(evaluations):
		if len(evaluations) >= 2 and evaluations[-2] - evaluations[-1] == 0 and evaluations[-1] < 10:
			return True
		else:
			return False

	replay_rates = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]
	data = []
	for rate in replay_rates:
		print(rate)
		#for i in range(10):
			#if i % 10 == 0:

		#print(f"iteration {i}:")
		start_time = time.process_time()
		agent = QAgent(environment = environment, learning_rate = 1., discount_factor=0.9, replay_rate = rate)

		num_episodes = 0
		num_iterations = 0
		goals_reached = 0

		evaluations = []
		goal_reached = False
		while (num_episodes < max_episodes and not (goal_reached and converge(evaluations))):
			goal, iterations = agent.episode(draw = args.draw)

			if goal:
				goals_reached += 1
				#print(f"{num_episodes:5d}:goal reached.")
			if num_episodes % 100 == 0:
				#pass
				print(f"{num_episodes:5d}:")


			if num_episodes > 0 and num_episodes % 100 == 0:
				goal_reached, iterations = agent.episode(draw = False, evaluate=True, max_iterations = 200)
				print("-"*20)
				print(f"evaluation:{goal_reached}")
				if goal_reached:
					print(f"iterations:{iterations}")
				print("-"*20)
				evaluations.append(iterations)
			num_episodes += 1

		goal, iterations = agent.episode(draw = False, evaluate=True, max_iterations = 200)

		data.append((rate, time.process_time() - start_time, iterations))
	# 		xs.append(rate)
	# 		ys.append(time.process_time()-start_time)

	# plt.scatter(xs, ys)

	# plt.show(block=True)
	# plt.savefig('times.png')

	with open('times.csv', 'a') as f:
		writer = csv.writer(f)

		for item in data:
			writer.writerow(item)




	# num_episodes = 0
	# num_iterations = 0
	# goals_reached = 0

	# evaluations = []
	# goal_reached = False
	# while (num_episodes < max_episodes and not (goal_reached and converge(evaluations))):
	# 	goal, iterations = agent.episode(draw = args.draw)

	# 	if goal:
	# 		goals_reached += 1
	# 		print(f"{num_episodes:5d}:goal reached.")
	# 	if num_episodes % 100 == 0:
	# 		print(f"{num_episodes:5d}:")


	# 	if num_episodes > 0 and num_episodes % 100 == 0:
	# 		goal_reached, iterations = agent.episode(draw = False, evaluate=True, max_iterations = 200)
	# 		print("-"*20)
	# 		print(f"evaluation:{goal_reached}")
	# 		if goal_reached:
	# 			print(f"iterations:{iterations}")
	# 		print("-"*20)
	# 		evaluations.append(iterations)


	# 	num_episodes += 1
	# episode_iterations = np.array(episode_iterations)

	# goal, iterations = agent.episode(draw = True, evaluate=True, max_iterations = 200)

	# print("-"*30)
	# print("Simulation ended.")
	# print(f"episodes   :{num_episodes}")
	# print(f"map solved :{goal}")
	# print(f"iterations :{iterations}")
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
	parser.add_argument('filename')
	parser.add_argument('--episodes', '-e', action='store', type=int, default=5000)
	parser.add_argument('--save_figure', '-s', action='store_true')
	parser.add_argument('--draw', '-d', action='store_true')
	args = parser.parse_args()
	main()