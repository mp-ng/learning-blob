import blob_env
from blob_env import BlobEnv
import modified_tensorboard
from modified_tensorboard import ModifiedTensorBoard

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

from collections import deque

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class DQNAgent:
	def __init__(self):

		# Main network for fitting / training
		self.model = self.create_model()

		# Target network for predicting
		self.target_model = self.create_model()
		self.target_model.set_weights(self.create_model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		self.tensorboard = ModifiedTensorBoard(
			log_dir=f"logs/{MODEL_NAME}-{(datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S'))}")
		self.target_update_counter = 0


	def create_model(self):
		model = keras.Sequential(
			[
				Conv2D(256, (3, 3), input_shape=(env.OBSERVATION_SPACE_VALUES)),
				Activation("relu"),
				MaxPooling2D(2, 2),
				Dropout(0.2),

				Conv2D(256, (3, 3)),
				Activation("relu"),
				MaxPooling2D(2, 2),
				Dropout(0.2),

				Flatten(),
				Dense(64),

				Dense(env.ACTION_SPACE_SIZE, activation="linear")
			]
		)
		model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["accuracy"])
		return model

	# Adds step's data to a memory replay array
	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	# Queries main network for Q values given current observation space (environment state)
	def get_qs(self, state, step):
		return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

	# Trains main network every step during episode
	def train(self, terminal_state, step):

		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		# Get a minibatch of random samples from memory replay array
		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		# Get current states from minibatch, then query NN model for Q values
		current_states = np.array([transition[0] for transition in minibatch]) / 255
		current_qs_list = self.model.predict(current_states)

		# Get future states from minibatch, then query NN model for Q values
		new_current_states = np.array([transition[3] for transition in minibatch]) / 255
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		# Enumerate our batches
		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			
			# If not a terminal state, get new q from future states, otherwise set it to 0
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			# Update Q value for given state
			current_qs = current_qs_list[index]
			current_qs[action] = new_q

		    # Append to our training data
			X.append(current_state)
			y.append(current_qs)

		# Fit on all samples as one batch, log only on terminal state
		self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE,
			verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

		# Update target network counter every episode
		if terminal_state:
			self.target_update_counter += 1

		# If counter reaches set value, update target network with weights of main network
		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights)
			self.target_update_counter = 0


