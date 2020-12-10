import os
import random
from collections import namedtuple

import numpy as np
import tensorflow as tf

import environment
from network import Network, ReplayBuffer

# from environment import MapType
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

State = namedtuple('State', ['player', 'boxes'])


class Agent():
	actions = [environment.UP, environment.RIGHT, environment.DOWN, environment.LEFT]

	def __init__(self, environment, seed=None, *args, **kwargs):
		self.environment = environment
		if seed:
			random.seed(seed)
			np.random.seed(seed)
			tf.random.set_random_seed(seed)

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

	def __init__(self, environment, learning_rate=1., discount_factor=0.95, replay_rate=0.2, verbose=False, *args,
				 **kwargs):
		# super()
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
		for box in self.environment.state[1:]:
			if tuple(box) in self.environment.storage:
				count += 1

		return count

	def reward(self, state, action, sokoban_map):
		box_pushing = sokoban_map[tuple(state[0] + action)] == environment.BOX and sokoban_map[
			tuple(state[0] + 2 * action)] == environment.EMPTY
		push_on_goal = box_pushing and (tuple(state[0] + 2 * action) in self.environment.storage)

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
			# print("reward for finishing puzzle")
			return 500.
		elif push_on_goal:
			return 50.
		elif box_pushing:
			return -0.5
		elif self.environment.is_deadlock():
			# print("deadlock reward")
			return -2
		else:
			return -1

	def get_actions(self, state, sokoban_map):
		'''
		Gets "viable" actions for the robot. i.e. one's that don't move into walls or deadlocks
		'''
		viable_actions = []
		for action in self.actions:
			state_hash = self.next_state(state, action, sokoban_map).tobytes()
			if state_hash in self.environment.deadlock_table and all(
					[self.environment.deadlock_table[state_hash][key] for key in
					 self.environment.deadlock_table[state_hash]]):
				continue

			if sokoban_map[tuple(state[0] + action)] != environment.WALL:
				viable_actions.append(action)

		return viable_actions

	def next_state(self, state, action, sokoban_map):
		map_location = sokoban_map[tuple(state[0] + action)]
		if map_location == environment.WALL:
			next_state = state
		elif map_location == environment.BOX and sokoban_map[tuple(state[0] + 2 * action)] != environment.EMPTY:
			next_state = state
		elif map_location == environment.BOX:
			next_state = np.copy(state)

			for i in range(len(next_state[1:])):
				if (next_state[i + 1] == state[0] + action).all():
					next_state[i + 1] = state[0] + 2 * action
			next_state[0] = state[0] + action
		else:
			next_state = np.copy(state)
			next_state[0] = state[0] + action

		return next_state

	def update(self, state, action, sokoban_map):
		# print(self.encode(state, action))
		if self.encode(state, action) not in self.qtable:
			self.qtable[self.encode(state, action)] = 0.

		next_state = self.next_state(state, action, sokoban_map)

		next_actions = self.get_actions(next_state, sokoban_map)
		for possible_action in next_actions:
			if self.encode(next_state, possible_action) not in self.qtable:
				self.qtable[self.encode(next_state, possible_action)] = 0.

		qmax = np.amax(
			np.array([self.qtable[self.encode(next_state, possible_action)] for possible_action in next_actions]))
		self.qtable[self.encode(state, action)] += self.learning_rate * (
				self.reward(state, action, sokoban_map) + self.discount_factor * qmax - self.qtable[
			self.encode(state, action)])

	# print(f"{self.encode(state, action)}:{self.qtable[self.encode(state, action)]}")

	def learn(self, state, sokoban_map):
		# exploration
		if random.random() < 0.99:  # greedy rate
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
				print(
					f"{environment.direction_to_str(possible_action)}:{self.qtable[self.encode(state, possible_action)]}")

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

		# print(f"chosen action:{chosen_action}")
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

			if evaluate:
				if (pstate_1 == self.environment.state).all() or (pstate_2 == self.environment.state).all():
					break

				pstate_2 = np.copy(pstate_1)
				pstate_1 = np.copy(self.environment.state)

			action_sequence.append(action)
			num_iterations += 1

		if self.environment.is_goal():
			self.experience_cache.append(action_sequence)

			if len(self.experience_cache) > 5:  # limit to 5 experiences
				self.experience_cache.pop(0)

		if not evaluate and self.experience_cache and random.random() < self.replay_rate:
			self.replay()

		return self.environment.is_goal(), num_iterations


class DeepQAgent(Agent):
	def __init__(self, environment, discount_factor=.95, learn_freq=4, target_update_freq=100, batch_size=32,
				 max_explore=1,
				 min_explore=0.05,
				 anneal_rate=(1 / 200),
				 replay_memory_size=100, seed=None, *args, **kwargs):
		super().__init__(environment, seed=seed, *args, **kwargs)

		state_size = np.size(self.environment.state, axis=0)
		self.position_one_hot = np.eye(self.environment.xlim * self.environment.ylim)

		self.online_network = Network(state_size * len(self.position_one_hot), len(self.actions))
		self.target_network = Network(state_size * len(self.position_one_hot), len(self.actions))

		self.update_target_network()

		self.batch_size = batch_size
		self.learn_frequent = learn_freq
		self.replay_memory_size = replay_memory_size
		self.target_update_frequent = target_update_freq
		self.steps = 0
		self.discount_factor = discount_factor
		self.max_explore = max_explore
		self.min_explore = min_explore
		self.anneal_rate = anneal_rate

		self.replay_buffer = ReplayBuffer(replay_memory_size)

	def policy(self, state, training):
		explore_prob = max(self.max_explore - (self.steps * self.anneal_rate), self.min_explore)
		explore = explore_prob > np.random.rand()
		if training and explore:
			action_idx = np.random.randint(len(self.actions))
		else:
			inputs = self.encode_position(self.encode_state(state))
			qvalues = self.online_network.predict(inputs)
			action_idx = np.argmax(qvalues, axis=-1)[0]

		return self.actions[action_idx]

	def reward(self, state, action, sokoban_map: np.ndarray):
		box_pushing = sokoban_map[tuple(state[0] + action)] == environment.BOX and sokoban_map[
			tuple(state[0] + 2 * action)] in (environment.EMPTY, environment.GOAL)
		push_on_goal = box_pushing and sokoban_map[tuple(state[0] + 2 * action)] == environment.GOAL

		num_of_boxes = len(state) - 1
		# push the last box on goal will reach goal
		goal_reach = push_on_goal and num_of_boxes - 1 == (sokoban_map == environment.BOX_IN_GOAL).sum()

		if goal_reach:
			print("reward for finishing puzzle")
			return 1.
		elif push_on_goal:
			print("push on goal")
			return 0.1
		elif box_pushing:
			return -0.001
		elif self.environment.is_deadlock():
			# print("deadlock reward")
			return -0.004
		else:
			return -0.002

	def append_reward(self, state, action, next_state, sokoban_map, terminated):
		action_idx = 0
		for i, a in enumerate(self.actions):
			if np.array_equal(a, action):
				action_idx = i
		reward = self.reward(state, action, sokoban_map)
		self.replay_buffer.add({
			'action': action_idx,
			'state': self.encode_state(state),
			'reward': reward,
			'next_state': self.encode_state(next_state),
			'continued': not terminated,
		})

	def episode(self, draw=False, evaluate=False, max_iterations=1500):
		self.environment.reset()
		num_iteration = 0
		should_terminated = False
		while not should_terminated and num_iteration < max_iterations:
			self.steps += 1

			sokoban_map = np.copy(self.environment.map)
			state = np.copy(self.environment.state)
			action = self.policy(self.environment.state, not evaluate)
			self.environment.step(action)
			next_state = np.copy(self.environment.state)
			should_terminated = self.environment.is_goal() or self.environment.is_deadlock()
			if draw:
				self.environment.draw()

			if not evaluate:
				self.append_reward(state, action, next_state, sokoban_map, should_terminated)

				if self.steps % self.learn_frequent == 0 and self.steps > self.replay_memory_size:
					self.train_network()
				if self.steps % self.target_update_frequent == 0:
					self.update_target_network()

			num_iteration += 1
		print("steps: %d / goal: %s dead: %s" % (
			num_iteration, self.environment.is_goal(), self.environment.is_deadlock()))
		return self.environment.is_goal(), self.steps

	def encode_state(self, state):
		return np.sum((state + [0, -1]) * [1, self.environment.xlim], axis=1)

	def encode_position(self, positions):
		one_hot = self.position_one_hot[positions]
		return one_hot.reshape((1, -1))

	def decode_state(self, arr):
		return np.array([
			((arr - 1) / self.environment.xlim + 1).astype(int),
			(arr - 1) % self.environment.xlim + 1
		]).reshape((-1, 2), order='F')

	def decode_position(self, one_hot):
		return one_hot.reshape((-1, len(self.position_one_hot)))

	def train_network(self):
		batch = self.replay_buffer.sample(self.batch_size)
		inputs = np.array([self.encode_position(b["state"]) for b in batch])
		actions = np.array([b["action"] for b in batch])
		rewards = np.array([b["reward"] for b in batch])
		next_inputs = np.array([self.encode_position(b["next_state"]) for b in batch])
		continued = np.array([b["continued"] for b in batch])
		actions_one_hot = np.eye(len(self.actions))[actions]

		next_qvalues = self.target_network.predict(np.squeeze(next_inputs))
		targets = rewards + (continued * self.discount_factor) * tf.reduce_max(next_qvalues, axis=1)

		return self.online_network.train_step(np.squeeze(inputs), targets, actions_one_hot)

	def update_target_network(self):
		self.target_network.model.set_weights(self.online_network.model.get_weights())

	def save(self, folder_name, **kwargs):
		"""Saves the Agent and all corresponding properties into a folder
		Arguments:
			folder_name: Folder in which to save the Agent
			**kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
		"""

		# Create the folder for saving the agent
		if not os.path.isdir(folder_name):
			os.makedirs(folder_name)

		# Save DQN and target DQN
		self.online_network.model.save_weights(folder_name + '/dqn.h5')
		self.target_network.model.save_weights(folder_name + '/target_dqn.h5')

		# Save replay buffer
		self.replay_buffer.save(folder_name + '/replay-buffer')

	def load(self, folder_name, load_replay_buffer=True):
		"""Load a previously saved Agent from a folder
		Arguments:
			folder_name: Folder from which to load the Agent
		Returns:
			All other saved attributes, e.g., frame number
		"""

		if not os.path.isdir(folder_name):
			raise ValueError(f'{folder_name} is not a valid directory')

		# Load DQNs
		self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
		self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
		self.optimizer = self.DQN.optimizer

		# Load replay buffer
		if load_replay_buffer:
			self.replay_buffer.load(folder_name + '/replay-buffer')
