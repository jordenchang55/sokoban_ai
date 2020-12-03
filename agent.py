import random
import random

import numpy as np

import environment
#from environment import MapType


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


	def __init__(self, environment, learning_rate = 1., discount_factor = 0.95, replay_rate = 0.2, verbose = False,*args, **kwargs):
		#super()
		super().__init__(environment, args, kwargs)
		self.qtable = {}

		self.experience_cache = []
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.replay_rate = replay_rate
		self.verbose = verbose

	def encode(self, state, action):
		return (state.tobytes(), action.tobytes())

	def count_box_on_goal(self):
		count = 0
		for box in self.environment.state[2:]:
			if tuple(box) in self.environment.storage:
				count += 1

		return count
	def reward(self, state, action, sokoban_map):
		box_position = tuple(state[0] + action)
		box_pushing = sokoban_map[box_position] == environment.BOX and sokoban_map[tuple(state[0] + 2*action)] == environment.EMPTY
		push_on_goal = box_pushing and (tuple(state[0]+2*action) in self.environment.storage)

		not_scored = False
		for i in range(len(self.environment.state[2:])):
			if (box_position == self.environment.state[2+i]).all():
				not_scored = not self.environment.has_scored[i]



		goal_reach = all([sokoban_map[place] == environment.BOX for place in self.environment.storage])
		if push_on_goal:
			goal_reach = True
			set_difference = self.environment.storage.difference({tuple(state[0] + 2 * action)})
			for place in set_difference:
				if sokoban_map[place] != environment.BOX:
					goal_reach = False
		else:
			goal_reach = False

		if goal_reach:
			#print("reward for finishing puzzle")
			return 500.
		elif push_on_goal and not_scored:
			return 50. 

		# elif box_pushing:
		# 	return -0.5
		# elif self.environment.is_deadlock():
		# 	#print("deadlock reward")
		# 	return -2
		else:
			return -1

	def get_actions(self, state, sokoban_map):
		'''
		Gets "viable" actions for the robot. i.e. one's that don't move into walls or deadlocks
		'''
		viable_actions = []
		for action in self.actions:
			next_state = self.next_state(state, action, sokoban_map)
			next_boxes_hash = next_state[2:].tobytes()
			if next_boxes_hash in self.environment.deadlock_table and all([self.environment.deadlock_table[next_boxes_hash][key] for key in self.environment.deadlock_table[next_boxes_hash]]):
				continue

			if sokoban_map[tuple(state[0] + action)] != environment.WALL:
				viable_actions.append(action)

		return viable_actions



	def next_state(self, state, action, sokoban_map):
		map_location = sokoban_map[tuple(state[0] + action)]
		if map_location == environment.WALL:
			next_state = state
		elif map_location == environment.BOX and sokoban_map[tuple(state[0] + 2*action)] != environment.EMPTY:
			next_state = state
		elif map_location == environment.BOX:
			next_state = np.copy(state)

			
			for i in range(len(next_state[2:])):
				if (next_state[i+2] == state[0]+action).all():
					next_state[i+2] = state[0] + 2*action 

					if self.environment.has_scored[i] == 0 and tuple(next_state[i+2]) in self.environment.storage:
						#if agent hasn't scored this box but is about to score the box
						next_state[1,0] += 1
					break
			next_state[0] = state[0] + action
		else:
			next_state = np.copy(state)
			next_state[0] = state[0] + action

		#print(next_state)
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

		if next_actions:
			qmax = np.amax(np.array([self.qtable[self.encode(next_state, possible_action)] for possible_action in next_actions]))
		else:
			qmax = -1.
		self.qtable[self.encode(state, action)] += self.learning_rate*(self.reward(state, action, sokoban_map) + self.discount_factor*qmax - self.qtable[self.encode(state, action)])

		#print(f"{self.encode(state, action)}:{self.qtable[self.encode(state, action)]}")


	def learn(self, state, sokoban_map):
		#exploration
		if random.random() < 0.20: #greedy rate
			chosen_action = random.choice(self.get_actions(state, sokoban_map))
		else:
			chosen_action = self.evaluate(state, sokoban_map)
		self.update(state, chosen_action, sokoban_map)
		return chosen_action
		
	def evaluate(self, state, sokoban_map):
		chosen_action = None
		chosen_value = 0.
		for possible_action in self.get_actions(state, sokoban_map):

			if self.verbose:
				print(f"{environment.direction_to_str(possible_action)}:{self.qtable[self.encode(state, possible_action)]}")
			
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
		pstate_1 = None
		pstate_2 = None
		while not self.environment.is_goal():
			if self.environment.is_deadlock():
				self.environment.undo()

			if not evaluate:
				action = self.learn(self.environment.state, self.environment.map)
			else:
				action = self.evaluate(self.environment.state, self.environment.map)
			self.environment.step(action)

			if draw:
				self.environment.draw()

			if num_iterations > max_iterations:
				break

			if evaluate:
				if (pstate_1 == self.environment.state).all() or (pstate_2 == self.environment.state).all():
					break

				pstate_2 = np.copy(pstate_1)
				pstate_1 = np.copy(self.environment.state)
				



			action_sequence.append(action)
			num_iterations += 1


		if self.environment.is_goal():
			self.experience_cache.append(action_sequence)

			if len(self.experience_cache) > 5: #limit to 5 experiences
				self.experience_cache.pop(0)

		if not evaluate and self.experience_cache and random.random() < self.replay_rate:
			self.replay()


		return self.environment.is_goal(), num_iterations