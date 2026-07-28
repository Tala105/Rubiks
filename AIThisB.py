import os
import random
from collections import deque
from math import radians
from uuid import uuid4

import numpy as np
import tensorflow as tf
import keras
import pyglet
from pyglet.gl import *
from pyglet.graphics import Batch
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

from cube import Cube
from cubePieces import Piece
from Constants import NUM_CUBES, SAVE_ITERATION, RESET_ITERATION, FPS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MOVES = ["U", "Ud", "R", "Rd", "F", "Fd", "D", "Dd", "L", "Ld", "B", "Bd"]
NUM_ACTIONS = len(MOVES)
UNDO_PAIRS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}
SOLVED_STATE = Cube().get_state()

COLOR_MAP = {
    'W': (1, 1, 1, 1), 'Y': (1, 1, 0, 1), 'R': (1, 0, 0, 1),
    'O': (1, 0.5, 0, 1), 'G': (0, 1, 0, 1), 'B': (0, 0, 1, 1),
    'X': (0.2, 0.2, 0.2, 1.0),
}

VERTEX_SOURCE = """#version 460
    in vec3 position;
    in vec4 colors;
    out vec4 vertex_colors;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);
        vertex_colors = colors;
    }
"""

FRAGMENT_SOURCE = """#version 460
    in vec4 vertex_colors;
    out vec4 final_color;

    void main()
    {
        if (vertex_colors.a == 0.0) discard;
        final_color = vertex_colors;
    }
"""

SHADER_PROGRAM = ShaderProgram(Shader(VERTEX_SOURCE, 'vertex'), Shader(FRAGMENT_SOURCE, 'fragment'))


class CubeWindow(pyglet.window.Window):
    def __init__(self, cube, index, size=(320, 240)):
        super().__init__(*size, resizable=True, caption=f"cube_{index}")
        self.cube = cube

    def on_draw(self):
        self.clear()
        batch = Batch()
        glEnable(GL_DEPTH_TEST)
        x_rot = radians(35.26)
        y_rot = radians(-45)
        self.projection = Mat4.orthogonal_projection(-3, 3, -3, 3, -255, 255)
        self.view = (
            Mat4.from_translation(Vec3(-2, 2, -20))
            @ Mat4.from_rotation(x_rot, Vec3(1, 0, 0))
            @ Mat4.from_rotation(y_rot, Vec3(0, 1, 0))
        )
        pieces = sum(self.cube._pieces, [])
        size = 0.25
        for i, piece in enumerate(pieces):
            x_off = (i % 3) * 2 * size
            y_off = -(i // 3 - 3 * (i // 9)) * 2 * size
            z_off = -(i // 9) * 2 * size
            self._draw_piece(piece, (x_off, y_off, z_off), size, batch)
        batch.draw()

    def _draw_piece(self, piece: Piece, offset, size, batch):
        s = size
        vertices = (
            [s, s, s], [s, -s, s], [-s, -s, s], [-s, s, s],
            [s, s, -s], [s, -s, -s], [-s, -s, -s], [-s, s, -s],
        )
        faces_index = (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (0, 3, 7, 4), (1, 2, 6, 5),
        )
        face_colors = {
            0: COLOR_MAP[piece.colors['F']],
            1: COLOR_MAP[piece.colors['B']],
            2: COLOR_MAP[piece.colors['R']],
            3: COLOR_MAP[piece.colors['L']],
            4: COLOR_MAP[piece.colors['T']],
            5: COLOR_MAP[piece.colors['D']],
        }
        for i, f in enumerate(faces_index):
            positions = []
            for index in f:
                vertex = [x + y for x, y in zip(vertices[index], offset)]
                positions.extend(vertex)
            SHADER_PROGRAM.vertex_list_indexed(
                4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
                position=('f', tuple(positions)),
                colors=('f', face_colors[i] * 4),
                batch=batch,
            )


def apply_move(cube, action, last_action, rewarded_indices, step_count, max_steps):
    getattr(cube, MOVES[action])()

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

    def reset_gamma(self):
        self.gamma = 0

    def save(self):
        self.model.save(self.model_path)

    def load(self, input_path=None):
        path = input_path if input_path else self.model_path
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
        else:
            print("No model to load")


def train(
    num_cubes: int = NUM_CUBES,
    num_iterations: int = 100_000,
    max_steps: int = 1000,
    batch_size: int = 128,
    train_steps_per_iteration: int = 10,
    target_update_every: int = 10,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    render_indices=None,
):
    cubes = [Cube() for _ in range(num_cubes)]
    for cube in cubes:
        cube.scramble()

    render_indices = set(render_indices or [])
    windows = [CubeWindow(cubes[i], i) for i in range(num_cubes) if i in render_indices]

    last_actions = [None] * num_cubes
    rewarded_indices = [set() for _ in range(num_cubes)]
    step_counts = [0] * num_cubes
    episode_rewards = np.zeros(num_cubes, dtype=np.float32)

    states = np.array([cube.get_state() for cube in cubes], dtype=np.float32)
    agent = DQNAgent(state_size=states.shape[1])

    epsilon = epsilon_start
    iteration = 0

    def step(dt):
        nonlocal epsilon, iteration
        if iteration >= num_iterations:
            pyglet.app.exit()
            return

        actions = agent.act_batch(states, epsilon)
        next_states = np.zeros_like(states)

        for i, cube in enumerate(cubes):
            next_state, reward, done = apply_move(
                cube, int(actions[i]), last_actions[i], rewarded_indices[i],
                step_counts[i], max_steps,
            )
            step_counts[i] += 1
            last_actions[i] = int(actions[i])
            episode_rewards[i] += reward

            if done:
                print(f"[cube {i}] episode reward={episode_rewards[i]:.2f} epsilon={epsilon:.3f}")
                cube.make_solved_cube()
                cube.scramble()
                next_state = np.array(cube.get_state(), dtype=np.float32)
                last_actions[i] = None
                rewarded_indices[i] = set()
                step_counts[i] = 0
                episode_rewards[i] = 0.0

            next_states[i] = next_state
            agent.remember(states[i], actions[i], reward, next_state, done)

        states[:] = next_states
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        for _ in range(train_steps_per_iteration):
            agent.replay(batch_size)

        if iteration % target_update_every == 0:
            agent.update_target()

        if iteration % SAVE_ITERATION == 0:
            agent.save()
            print(f"Iteration {iteration} complete (epsilon={epsilon:.3f})")
        
        if iteration % RESET_ITERATION == 0:
            agent.reset_gamma()
        
        iteration += 1

    pyglet.clock.schedule_interval(step, 1 / FPS)
    pyglet.app.run()


if __name__ == "__main__":
    # train(render_indices=range(NUM_CUBES))
    train(render_indices=[-1])
