import random
import random

import numpy as np

import environment
from environment import MapType


from collections import namedtuple
State = namedtuple('State', ['player', 'boxes'])

class Agent():
	actions = [environment.UP, environment.RIGHT, environment.DOWN, environment.LEFT]

	def __init__(self, environment, *args, **kwargs):
		self.environment = environment


	def learn(self, state, sokoban_map):

		return random.choice(self.actions)


	def evaluate(self, state, sokoban_map):
		return random.choice(self.actions)




class SimpleAgent(Agent):

	def __init__(self, *args, **kwargs):
		pass

	def heuristic(self, state, sokoban_map):
		pass


	def get_action(self, state, sokoban_map):


		pass	




class QAgent(Agent):


	def __init__(self, environment, *args, **kwargs):
		#super()
		super().__init__(environment, args, kwargs)
		self.qtable = {}

		self.experience_cache = []
		self.learning_rate = kwargs['learning_rate']
		self.discount_factor = kwargs['discount_factor']
		self.replay_rate = kwargs['replay_rate']

	def encode(self, state, action):
		return (state.tobytes(), action.tobytes())


	def reward(self, state, action, sokoban_map):
		box_pushing = sokoban_map[tuple(state[0] + action)] == MapType.BOX.value and sokoban_map[tuple(state[0] + 2*action)] == MapType.EMPTY.value
		push_on_goal = box_pushing and (tuple(state[0]+2*action) in self.environment.storage)

		# goal_reach = all([sokoban_map[place] == MapType.BOX.value for place in self.environment.storage])
		if push_on_goal:
			goal_reach = True
			set_difference = self.environment.storage.difference({tuple(state[0] + 2 * action)})
			for place in set_difference:
				if sokoban_map[place] != MapType.BOX.value:
					goal_reach = False
		else:
			goal_reach = False

		if goal_reach:
			#print("reward for finishing puzzle")
			return 500.
		elif push_on_goal:
			return 50.
		else:
			return -1


	def get_actions(self, state, sokoban_map):
		'''
		Gets "viable" actions for the robot. i.e. one's that don't move into walls
		'''
		viable_actions = []
		for action in self.actions:
			if sokoban_map[tuple(state[0] + action)] != MapType.WALL.value:
				viable_actions.append(action)

		return viable_actions



	def next_state(self, state, action, sokoban_map):
		map_location = sokoban_map[tuple(state[0] + action)]
		if map_location == MapType.WALL.value:
			next_state = state
		elif map_location == MapType.BOX.value and sokoban_map[tuple(state[0] + 2*action)] != MapType.EMPTY.value:
			next_state = state
		elif map_location == MapType.BOX.value:
			next_state = np.copy(state)

			
			for i in range(len(next_state[1:])):
				if (next_state[i+1] == state[0]+action).all():
					next_state[i+1] = state[0] + 2*action 
			next_state[0] = state[0] + action
		else:
			next_state = np.copy(state)
			next_state[0] = state[0] + action

		return next_state

	def update(self, state, action, sokoban_map):
		#print(self.encode(state, action))
		if self.encode(state, action) not in self.qtable:
			self.qtable[self.encode(state, action)] = 0.

		next_state = self.next_state(state, action, sokoban_map)

		next_actions = self.get_actions(next_state, sokoban_map)
		for possible_action in next_actions:
			if self.encode(next_state, possible_action) not in self.qtable:
				self.qtable[self.encode(next_state, possible_action)] = 0.

		qmax = np.amax(np.array([self.qtable[self.encode(next_state, possible_action)] for possible_action in next_actions]))
		self.qtable[self.encode(state, action)] += self.learning_rate*(self.reward(state, action, sokoban_map) + self.discount_factor*qmax - self.qtable[self.encode(state, action)])

		#print(f"{self.encode(state, action)}:{self.qtable[self.encode(state, action)]}")


	def learn(self, state, sokoban_map):
		#exploration
		if random.random() < 1.:
			chosen_action = random.choice(self.get_actions(state, sokoban_map))
		else:
			chosen_action = self.evaluate(state, sokoban_map)
		self.update(state, chosen_action, sokoban_map)
		return chosen_action
		
	def evaluate(self, state, sokoban_map):
		chosen_action = None
		chosen_value = 0.
		for possible_action in self.get_actions(state, sokoban_map):

			
			if self.encode(state, possible_action) not in self.qtable:
				self.qtable[self.encode(state, possible_action)] = 0. #represents an unseen state... not ideal while evaluating

			# print(possible_action)
			# print(self.qtable[self.encode(state, possible_action)])

			if chosen_action is None:
				chosen_action = possible_action
				chosen_value = self.qtable[self.encode(state, possible_action)]
			else:
				potential_value = self.qtable[self.encode(state, possible_action)]
				if chosen_value < potential_value:
					#keep this one
					chosen_action = possible_action
					chosen_value = potential_value

		#print(f"chosen action:{chosen_action}")
		return chosen_action

	def replay(self):
		# choose experience from experience cache

		experience = random.choice(self.experience_cache)
		self.environment.reset()
		for action in experience:
			self.update(self.environment.state, action, self.environment.map)
			# if random.random() < 0.05:
			# 	action = self.learn(State(player=self.environment.player, boxes=self.environment.boxes), self.environment.map)
			# else:
			# 	action = random.choice(self.actions)
			self.environment.step(action)



	def episode(self, draw=False, evaluate=False, max_iterations=1500):
		action_sequence = []
	
		self.environment.reset()

		num_iterations = 0
		while not self.environment.is_goal() and not self.environment.is_deadlock():
			if not evaluate:
				action = self.learn(self.environment.state, self.environment.map)
			else:
				action = self.evaluate(self.environment.state, self.environment.map)
			self.environment.step(action)

			if draw:
				self.environment.draw()

			if num_iterations > max_iterations:
				break

			action_sequence.append(action)
			num_iterations += 1


		if self.environment.is_goal():
			self.experience_cache.append(action_sequence)

			if len(self.experience_cache) > 5: #limit to 5 experiences
				self.experience_cache.pop(0)

		if self.experience_cache and random.random() < self.replay_rate:
			self.replay()


		return self.environment.is_goal(), num_iterations