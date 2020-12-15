import random

import numpy as np

from environment import Environment


class Agent:
	actions = [Environment.UP, Environment.RIGHT, Environment.DOWN, Environment.LEFT]

	def __init__(self, environment, quiet, verbose):
		self.environment = environment
		self.quiet = quiet
		self.verbose = verbose and not quiet  # quiet outweighs verbose

		self.print_threshold = 200

		self.num_episodes = 0
		self.num_iterations = 0

	def verbose_print(self, string):
		if self.verbose:
			print(string)

	def standard_print(self, string):
		if not self.quiet:
			print(string)

	def episode_print(self, string=""):

		self.verbose_print(f"{self.num_episodes:6d}.{self.num_iterations:5d}:{string}")

	def reward(self, state, action):
		raise NotImplementedError

	def learn(self, state):

		raise NotImplementedError

	def evaluate(self, state):
		raise NotImplementedError

	def episode(self):
		raise NotImplementedError


# class SpeedyQAgent(Agent):


#   def __init__(self, environment, learning_rate = 1., discount_factor=09.5, replay_rate=0.2, verbose=False, *args, **kwargs):
#       super().__init__(environment, args, kwargs)

#       self.qtable = {}

#       self.experience_cache = []
#       self.learning_rate = learning_rate
#       self.discount_factor = discount_factor
#       self.replay_rate = replay_rate
#       self.verbose = verbose

#       self.inspiration = []


#   def learn(self, state, sokoban_map):


#       if state not in qtable:
#           qtable[state] = np.zeros((self.environment.xlim, self.environment.ylim, len(self.actions)))


#       qmax = [[ for j in range(self.environment.ylim)] for i in range(self.environment.xlim)]
#       qtable[state] = self.learning_rate*(self.reward(state, action, sokoban_map) + self.discount_factor*qmax - self.qtable[self.encode(state, action)])


class QAgent(Agent):

	def __init__(self, environment, discount_factor=0.95, learning_rate=1e-6, verbose=False, *args, **kwargs):
		# super()
		super().__init__(environment, args, kwargs)
		self.qtable = {}

		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.verbose = verbose

		self.num_episodes = 0

		self.q_sequence = []

	def encode(self, state, action):
		return (state.map.tobytes(), action.tobytes())

	def reward(self, state, action):
		box_position = np.array(self.environment.get_player(state)) + action
		next_box_position = box_position + action

		box_pushing = state.map[tuple(box_position)] == state.BOX and state.map[tuple(next_box_position)] == state.EMPTY
		push_on_goal = box_pushing and (tuple(next_box_position) in self.environment.storage)

		# not_scored = False
		# for i in range(len(self.environment.state[2:])):
		#   if (box_position == self.environment.state[2+i]).all():
		#       not_scored = not self.environment.has_scored[i]

		if push_on_goal:
			goal_reach = True
			set_difference = self.environment.storage.difference({tuple(next_box_position)})
			for place in set_difference:
				if state.map[tuple(place)] != state.BOX:
					goal_reach = False
		else:
			goal_reach = False

		state_hash = state.map.tobytes()
		if goal_reach:
			# print("reward for finishing puzzle")
			return 500.
		elif push_on_goal:
			# next_state = self.next_state(state, action, sokoban_map)
			# self.inspiration.append((copy.deepcopy(state), copy.deepcopy(self.environment.has_scored)))
			return 50.  # .
		elif state_hash in self.environment.deadlock_table and any(
				[self.environment.deadlock_table[state_hash][key] for key in
				 self.environment.deadlock_table[state_hash]]):
			# print('deadlock reward')
			return -5.
		# elif box_pushing:
		#   return -0.5
		# elif self.environment.is_deadlock():
		#   #print("deadlock reward")
		#   return -2
		else:
			return -1.

	def get_actions(self, state):
		'''
		Gets "viable" actions for the robot. i.e. one's that don't move into walls or deadlocks
		'''
		viable_actions = []

		for action in self.actions:

			# state_hash = state.map.tobytes()
			# if next_boxes_hash in self.environment.deadlock_table and any([self.environment.deadlock_table[next_boxes_hash][key] for key in self.environment.deadlock_table[next_boxes_hash]]):
			#   continue
			next_position = self.environment.get_player(state) + action
			if self.environment.state.map[tuple(next_position)] != state.WALL:
				viable_actions.append(action)

			state = self.environment.next_state(state, action)

		return viable_actions

	def get_greedy_rate(self):
		pass

	def update(self, state, action):
		# print(self.encode(state, action))
		if self.encode(state, action) not in self.qtable:
			self.qtable[self.encode(state, action)] = 0.

		next_state = self.environment.next_state(state, action)

		next_actions = self.get_actions(next_state)
		for possible_action in next_actions:
			if self.encode(next_state, possible_action) not in self.qtable:
				self.qtable[self.encode(next_state, possible_action)] = 0.

		if next_actions:
			qmax = np.amax(
				np.array([self.qtable[self.encode(next_state, possible_action)] for possible_action in next_actions]))
		else:
			qmax = -1.
		self.qtable[self.encode(state, action)] += self.learning_rate * (
				self.reward(state, action) + self.discount_factor * qmax - self.qtable[self.encode(state, action)])

	# print(f"{self.encode(state, action)}:{self.qtable[self.encode(state, action)]}")

	def learn(self, state):
		# exploration
		if random.random() < 0.2:  # greedy rate
			possible_actions = self.get_actions(state)
			# have_seen = [self.encode(state, possible_action) in self.qtable for possible_action in possible_actions]
			# if all(have_seen):
			chosen_action = random.choice(possible_actions)
		# else:
		#   chosen_action = possible_actions[0]
		#   for index, seen_action in enumerate(have_seen):
		#       if not seen_action:
		#           chosen_action = possible_actions[index]

		else:
			chosen_action = self.evaluate(state)
		self.update(state, chosen_action)
		return chosen_action

	def evaluate(self, state):
		chosen_action = None
		chosen_value = 0.

		for possible_action in self.get_actions(state):

			if self.verbose:
				print(
					f"{self.environment.direction_to_str(possible_action)}:{self.qtable[self.encode(state, possible_action)]}")

			if self.encode(state, possible_action) not in self.qtable:
				self.qtable[self.encode(state,
										possible_action)] = 0.  # represents an unseen state... not ideal while evaluating

			# print(possible_action)
			# print(self.qtable[self.encode(state, possible_action)])

			if chosen_action is None:
				chosen_action = possible_action
				chosen_value = self.qtable[self.encode(state, possible_action)]
			else:
				potential_value = self.qtable[self.encode(state, possible_action)]
				if chosen_value < potential_value:
					# keep this one
					chosen_action = possible_action
					chosen_value = potential_value

		self.q_sequence.append(chosen_value)
		# print(f"chosen action:{chosen_action}")
		return chosen_action

	# def replay(self):
	#     # choose experience from experience cache

	#     experience = random.choice(self.experience_cache)
	#     self.environment.reset()
	#     for action in experience:
	#         self.update(self.environment.state, action, self.environment.map)
	#         # if random.random() < 0.05:
	#         #   action = self.learn(State(player=self.environment.player, boxes=self.environment.boxes), self.environment.map)
	#         # else:
	#         #   action = random.choice(self.actions)
	#         self.environment.step(action)

	def episode(self, draw=False, evaluate=False, max_iterations=8000):
		action_sequence = []
		self.q_sequence = []

		state = self.environment.state.copy()

		print(f"{self.num_episodes:5d}.{0:7d}:")
		# if not evaluate and self.inspiration and random.random() < 0.1:
		#     inspired_state, inspired_scores = random.choice(self.inspiration)
		#     self.environment.state = inspired_state
		#     self.environment.has_scored = inspired_scores
		#     self.environment.reset_map()

		# draw = True

		self.num_iterations = 0
		pstate_1 = None
		pstate_2 = None
		goal_flag = False
		while self.num_iterations < max_iterations:
			if self.environment.is_goal_state(state):
				goal_flag = True
				break
			if self.environment.is_deadlock(state):
				break
			if not evaluate:
				action = self.learn(state)
			else:
				action = self.evaluate(state)
			self.environment.step(action)

			if draw:
				self.environment.draw()

			state = self.environment.next_state(state, action)
			if evaluate:
				if (pstate_1.map == self.environment.state.map).all() or (
						pstate_2.map == self.environment.state.map).all():
					break

				pstate_2 = pstate_1.copy()
				pstate_1 = state.copy()

			if self.num_iterations > 0 and self.num_iterations % self.print_threshold == 0:
				print(f"     .{self.num_iterations:7d}:")

			action_sequence.append(action)
			self.num_iterations += 1

		if self.num_iterations % self.print_threshold != 0:
			print(f"     .{self.num_iterations:7d}:")
		# if self.environment.is_goal():
		#   self.experience_cache.append(action_sequence)

		#   if len(self.experience_cache) > 5: #limit to 5 experiences
		#       self.experience_cache.pop(0)

		# if not evaluate and self.experience_cache and random.random() < self.replay_rate:
		#   self.replay()

		self.num_episodes += 1

		if evaluate:
			print("-" * 20)
			print(f"evaluation :{goal_flag}")
			if self.q_sequence:
				print(f"mean q(s,a):{np.array(self.q_sequence).mean():.4f}")
			if goal_flag:
				print(f"iterations :{self.num_iterations}")
			print("-" * 20)

		return self.environment.is_goal(), self.num_iterations
