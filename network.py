import random
from collections import deque

import numpy as np
import tensorflow as tf

random.seed(2)
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
				 weights_initializer=tf.initializers.glorot_uniform(),
				 bias_initializer=tf.initializers.zeros(),
				 optimizer=tf.keras.optimizers.Adam):
		if hidden_sizes is None:
			hidden_sizes = [50, 50]
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_sizes = hidden_sizes
		self.optimizer = optimizer()
		self._init_weight(weights_initializer, bias_initializer)

	def _init_weight(self, weights_initializer, bias_initializer):
		w_shape = []
		b_shape = []
		sizes = [self.input_size, *self.hidden_sizes, self.output_size]
		for i in range(1, len(sizes)):
			w_shape.append([sizes[i - 1], sizes[i]])
			b_shape.append([1, sizes[i]])
		self.weights = [init_weights(s, weights_initializer) for s in w_shape]
		self.biases = [init_weights(s, bias_initializer) for s in b_shape]

		self.trainable_variables = self.weights + self.biases

	def get_variable(self):
		return self.weights, self.biases

	def set_variables(self, weights, biases):
		self.weights = np.copy(weights)
		self.biases = np.copy(biases)
		self.trainable_variables = self.weights + self.biases

	def predict(self, inputs):
		prev_h = dense(inputs.astype(np.single), self.weights[0], self.biases[0], tf.nn.relu)
		n = len(self.hidden_sizes)
		for i in range(1, n):
			prev_h = dense(prev_h, self.weights[1], self.biases[1], tf.nn.relu)

		out = dense(prev_h, self.weights[n], self.biases[n])
		return out

	def train_step(self, inputs, targets, actions_one_hot):
		with tf.GradientTape() as tape:
			qvalues = tf.squeeze(self.predict(inputs))
			preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
			loss = tf.keras.losses.MSE(targets, preds)
			grads = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


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
