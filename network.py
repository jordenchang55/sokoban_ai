import csv
from collections import deque

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def dense(x, weights, bias, activation=tf.identity, **activation_kwargs):
	"""Dense layer."""
	z = tf.matmul(x, weights) + bias
	return activation(z, **activation_kwargs)


def init_weights(shape, initializer):
	"""Initialize weights for tensorflow layer."""
	weights = tf.Variable(
		initializer(shape),
		trainable=True,
		dtype=tf.float32
	)

	return weights


class Network:
	def __init__(self, input_size, output_size, hidden_sizes=None,
				 optimizer=tf.train.AdamOptimizer):
		if hidden_sizes is None:
			hidden_sizes = [50, 50]
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(input_size))
		for hidden in hidden_sizes:
			self.model.add(tf.keras.layers.Dense(hidden, activation='relu'))
		self.model.add(tf.keras.layers.Dense(output_size))
		self.model.compile(optimizer(learning_rate=0.00001), loss='mse')

	def predict(self, inputs):
		return self.model.predict(inputs.astype(np.single))

	def train_step(self, inputs, targets, actions_one_hot):
		with tf.GradientTape() as tape:
			qvalues = self.model(inputs.astype(np.single))
			preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
			loss = tf.keras.losses.MSE(targets, preds)
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return loss


class ReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer = deque(maxlen=buffer_size)

	def __len__(self):
		return len(self.buffer)

	def add(self, exp):
		self.buffer.append(exp)

	def sample(self, sample_size):
		buffer_size = len(self.buffer)
		index = np.random.choice(
			np.arange(buffer_size),
			size=sample_size,
			replace=False
		)
		return [self.buffer[i] for i in index]

	def save(self, filepath):
		with open(filepath, 'w') as f:
			writer = csv.DictWriter(f, fieldnames=['action', 'state', 'reward', 'next_state'])
			writer.writeheader()
			for exp in self.buffer:
				writer.writerow(exp)

	def load(self, filepath):
		with open(filepath, 'r') as f:
			reader = csv.DictReader(f, fieldnames=['action', 'state', 'reward', 'next_state'])
			for row in reader:
				exp = {
					'action': int(row['action']),
					'reward': float(row['reward']),
					'state': np.fromstring(row['state'][1:len(row['state']) - 1], dtype=int, sep=" "),
					'next_state': np.fromstring(row['next_state'][1:len(row['next_state']) - 1], dtype=int, sep=" ")
				}
				self.buffer.append(exp)
