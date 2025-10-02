import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Inisialisasi lingkungan OpenAI Gym (misalnya, CartPole)
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Membangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(state_size,), activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))

# Fungsi memilih aksi berdasarkan eksplorasi dan eksploitasi
def select_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # Eksplorasi
    q_values = model.predict(state)
    return np.argmax(q_values[0])  # Eksploitasi

# Proses training
for episode in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break

    # Update jaringan saraf (Training)
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

    # Kurangi epsilon (exploration decay)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay