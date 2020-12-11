import argparse
import csv
# import matplotlib.patches as patches
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agent import QAgent, DeepQAgent
from environment import Environment

# from enum import Enum

# from environment import MapType

# def debug_logging.info(fmt_string):
# 	logging.info(fmt_string)
# control variables
iteration_max = 1000  # deadlock by iteration

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def main():
	max_episodes = abs(args.episodes)

	filepath = Path(args.filename)
	# logging.info(args.filename)
	if not filepath.exists():
		raise ValueError("Path does not exist.")
	if not filepath.is_file():
		raise ValueError("Path is not a valid file.")

	with open(filepath, 'r') as file:
		csv_input = csv.reader(file, delimiter=' ')

		for index, row in enumerate(csv_input):

			def unpack(points):

				return [tuple([int(points[index + 1]), int(points[index])]) for index in range(1, len(points), 2)]

			# logging.info(index, row)

			if index == 0:
				# sizeH, sizeV

				xlim = int(row[0])
				ylim = int(row[1])
			if index == 1:
				# logging.info(MapType.WALL.value)
				walls = unpack(row)
			elif index == 2:
				boxes = unpack(row)
			elif index == 3:
				storage = unpack(row)
			elif index == 4:
				player = np.array([int(row[0]), int(row[1])])

	environment = Environment(walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim)
	if args.agent == 'dqn':
		estimated_complex = environment.ylim * environment.xlim * len(boxes) * 6
		agent = DeepQAgent(
			environment, seed=3,
			batch_size=int(0.7 * estimated_complex),
			replay_memory_size=estimated_complex,
			anneal_rate=(1 / 100000))
	else:
		agent = QAgent(environment)

	episode_bookmarks = []
	episode_iterations = []

	num_episodes = 0
	num_iterations = 0
	goals_reached = 0
	while num_episodes < max_episodes:
		goal, iterations = agent.episode(draw=args.draw)
		if goal:
			goals_reached += 1
			episode_bookmarks.append(num_episodes)
			episode_iterations.append(iterations)
		# logging.info(f"{num_episodes:5d}:goal reached.")
		if num_episodes % 100 == 0:
			logging.info(f"{num_episodes:5d}:")

		if num_episodes > 0 and num_episodes % 1000 == 0:
			goal, iterations = agent.episode(draw=True, evaluate=True, max_iterations=200)
			if args.save_network:
				map_name = args.filename.split("/")[-1].split(".")[0]
				agent.save("outputs/%s_%d_%s" % (map_name, num_episodes, goal))
			logging.info("-" * 20)
			logging.info(f"evaluation:{goal}")
			if goal:
				logging.info(f"iterations:{iterations}")
			logging.info("-" * 20)

		num_episodes += 1
	episode_iterations = np.array(episode_iterations)

	goal, iterations = agent.episode(draw=True, evaluate=True, max_iterations=200)

	logging.info("-" * 30)
	logging.info("Simulation ended.")
	logging.info(f"episodes   :{num_episodes}")
	logging.info(f"map solved :{goal}")
	logging.info(f"iterations :{iterations}")

	plt.show(block=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
	parser.add_argument('filename')
	parser.add_argument('--episodes', '-e', action='store', type=int, default=5000)
	parser.add_argument('--save_figure', '-s', action='store_true')
	parser.add_argument('--save_network', '-S', action='store_true')
	parser.add_argument('--draw', '-d', action='store_true')
	parser.add_argument('--agent', '-a', default='q-learning', choices=['q-learning', 'dqn'])
	parser.add_argument('--verbose', '-v', action='store_true')
	args = parser.parse_args()
	main()
