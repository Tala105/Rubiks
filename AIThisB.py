import os
import random
from collections import deque
from uuid import uuid4

import numpy as np
import tensorflow as tf
import keras

from cube import Cube
from cubeRender import CubeRenderer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MOVES = ["U", "Ud", "R", "Rd", "F", "Fd", "D", "Dd", "L", "Ld", "B", "Bd"]
NUM_ACTIONS = len(MOVES)
UNDO_PAIRS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}
SOLVED_STATE = Cube().get_state()


def apply_move(cube, action, last_action, rewarded_indices, step_count, max_steps, renderer=None):
    getattr(cube, MOVES[action])()
    if renderer is not None:
        renderer.handle_input()
        renderer.render()

    reward = -0.1
    if last_action is not None and UNDO_PAIRS[action] == last_action:
        reward -= 1.0

    state = cube.get_state()
    for i, (s, sol) in enumerate(zip(state, SOLVED_STATE)):
        if s == sol and i not in rewarded_indices:
            reward += 0.5
            rewarded_indices.add(i)

    done = state == SOLVED_STATE or step_count + 1 >= max_steps
    if state == SOLVED_STATE:
        reward += 100.0
        print("=" * 25 + "Cube solved!" + "=" * 25)

    return np.array(state, dtype=np.float32), reward, done


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        num_actions: int = NUM_ACTIONS,
        gamma: float = 0.95,
        learning_rate: float = 1e-3,
        buffer_size: int = 100_000,
        model_path: str = "models/checkpoint.keras",
    ):
        self.id = uuid4()
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.replay_buffer = deque(maxlen=buffer_size)
        self.model_path = model_path
        os.makedirs("models", exist_ok=True)
        self.save_path = f"models/{self.id}.keras"

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.load()
        self.update_target()
        self.model.summary()

    def _build_model(self) -> keras.Sequential:
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,), dtype="float32"),
        ])
        for _ in range(4):
            model.add(keras.layers.Dense(128))
            model.add(keras.layers.LeakyReLU(alpha=0.01))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(self.num_actions))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act_batch(self, states: np.ndarray, epsilon: float) -> np.ndarray:
        q_values = self.model(states, training=False).numpy()
        greedy_actions = np.argmax(q_values, axis=1)
        random_actions = np.random.randint(0, self.num_actions, size=len(states))
        explore_mask = np.random.rand(len(states)) < epsilon
        return np.where(explore_mask, random_actions, greedy_actions)

    def replay(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return None

        minibatch = random.sample(self.replay_buffer, batch_size)
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = np.array([t[1] for t in minibatch], dtype=np.int32)
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = np.array([t[4] for t in minibatch], dtype=np.float32)

        return self._train_step(states, actions, rewards, next_states, dones)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        action_masks = tf.one_hot(actions, self.num_actions)
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            selected_q = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - selected_q))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self):
        self.model.save(self.model_path)

    def load(self, input_path=None):
        path = input_path if input_path else self.model_path
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
        else:
            print("No model to load")


def train(
    num_cubes: int = 8,
    num_iterations: int = 10_000,
    max_steps: int = 250,
    batch_size: int = 64,
    train_steps_per_iteration: int = 10,
    target_update_every: int = 10,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    render_index: int = None,
):
    cubes = [Cube() for _ in range(num_cubes)]
    for cube in cubes:
        cube.scramble()

    renderer = CubeRenderer([cubes[render_index]]) if render_index is not None else None
    last_actions = [None] * num_cubes
    rewarded_indices = [set() for _ in range(num_cubes)]
    step_counts = [0] * num_cubes
    episode_rewards = np.zeros(num_cubes, dtype=np.float32)

    states = np.array([cube.get_state() for cube in cubes], dtype=np.float32)
    agent = DQNAgent(state_size=states.shape[1])
    epsilon = epsilon_start

    for iteration in range(num_iterations):
        actions = agent.act_batch(states, epsilon)
        next_states = np.zeros_like(states)
        rewards = np.zeros(num_cubes, dtype=np.float32)
        dones = np.zeros(num_cubes, dtype=np.float32)

        for i, cube in enumerate(cubes):
            r = renderer if i == render_index else None
            state, reward, done = apply_move(
                cube, int(actions[i]), last_actions[i], rewarded_indices[i],
                step_counts[i], max_steps, r,
            )
            step_counts[i] += 1
            last_actions[i] = int(actions[i])
            episode_rewards[i] += reward

            if done:
                print(f"[cube {i}] episode reward={episode_rewards[i]:.2f} epsilon={epsilon:.3f}")
                cubes[i] = Cube()
                cubes[i].scramble()
                if i == render_index:
                    renderer.cube_instance[0] = cubes[i]
                state = np.array(cubes[i].get_state(), dtype=np.float32)
                last_actions[i] = None
                rewarded_indices[i] = set()
                step_counts[i] = 0
                episode_rewards[i] = 0.0

            next_states[i] = state
            rewards[i] = reward
            dones[i] = done
            agent.remember(states[i], actions[i], reward, next_states[i], done)

        states = next_states
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        for _ in range(train_steps_per_iteration):
            agent.replay(batch_size)

        if iteration % target_update_every == 0:
            agent.update_target()

        if iteration % 100 == 0:
            agent.save()
            print(f"Iteration {iteration} complete (epsilon={epsilon:.3f})")


if __name__ == "__main__":
    train(render_index=0)
