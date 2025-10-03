import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import numpy as np
from cube import Cube
from cubeRender import CubeRenderer
from time import sleep


solved_state = Cube().get_state()

class FakeRenderer:
    def render(self):
        pass

class CustomEnv(py_environment.PyEnvironment):
    def __init__(self, render: bool = False):
        super().__init__()
        self.cube = Cube()
        if render:
            self.renderer = CubeRenderer([self.cube], display=(800, 600), window_pos=(1920, 0))
        else:
            self.renderer = FakeRenderer()
        sleep(3)
        self.cube.scramble()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=11)
        self._observation_spec = array_spec.ArraySpec(shape=(len(self.cube.get_state()),), dtype=np.int32)
        self._episode_ended = False
        self._step_count = 0
        self._last_action = None

    @property
    def moves(self):
        return [
        self.cube.U, self.cube.Ud,
        self.cube.R, self.cube.Rd,
        self.cube.F, self.cube.Fd,
        self.cube.D, self.cube.Dd,
        self.cube.L, self.cube.Ld,
        self.cube.B, self.cube.Bd
        ]
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.cube.scramble()
        self._episode_ended = False
        self._step_count = 0
        return ts.restart(np.array(self.cube.get_state(), dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        self.moves[action]()
        self.renderer.render()
        
        reward = -0.1
        undo_pairs = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6, 8:9, 9:8, 10:11, 11:10}
        if self._last_action is not None and undo_pairs[action] == self._last_action:
            reward += -1.0
            
        state = self.cube.get_state()
        if not hasattr(self, "_rewarded_indices"):
            self._rewarded_indices = set()
        for i, (s, sol) in enumerate(zip(state, solved_state)):
            if s == sol and i not in self._rewarded_indices:
                reward += 0.5
                self._rewarded_indices.add(i)
        
        if state == solved_state:
            reward += 100.0
            print("="*25 + "Cube solved!" + "="*25)
            self._episode_ended = True
        elif self._step_count >= 250:
             self._episode_ended = True
            
        self._step_count += 1
        if self._episode_ended:
            return ts.termination(np.array(state, dtype=np.int32), reward)
        else:
            return ts.transition(np.array(state, dtype=np.int32), reward, discount=0.99)

env: CustomEnv = CustomEnv()
tf_env = tf_py_environment.TFPyEnvironment(env)

q_net = q_network.QNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=(100,))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network=q_net,
                           optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=tf.Variable(0))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_env.batch_size,
                                                               max_length=100000)

def collect_data(env, policy, num_episodes=50):
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step


@tf.function
def train_step(experience):
    return agent.train(experience).loss

for _ in range(1000):
    collect_data(tf_env, agent.collect_policy, num_episodes=10)
    print(f"Training iteration {_}")
    for __ in range(10):
        experience, _ = replay_buffer.get_next(sample_batch_size=64, num_steps=2)
        train_loss = train_step(experience)
        print(f"Step {__}: loss = {train_loss.numpy()}")

