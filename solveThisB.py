import numpy as np
import keras
import pyglet

from cube import Cube
from AIThisB import MOVES, SOLVED_STATE, CubeWindow

MODEL_PATH = "models/checkpoint.keras"


def solve_step(cube, model):
    state = np.array([cube.get_state()], dtype=np.float32)
    q_values = model(state, training=False).numpy()[0]
    action = int(np.argmax(q_values))
    getattr(cube, MOVES[action])()
    return cube.get_state() == SOLVED_STATE


def test(num_cubes=8, max_steps=250, render_indices=None, model_path=MODEL_PATH):
    model = keras.models.load_model(model_path)

    cubes = [Cube() for _ in range(num_cubes)]
    for cube in cubes:
        cube.scramble()

    render_indices = set(render_indices or [])
    windows = [CubeWindow(cubes[i], i) for i in range(num_cubes) if i in render_indices]

    step_counts = [0] * num_cubes
    solved = [False] * num_cubes
    finished = [False] * num_cubes

    def step(dt):
        if all(finished):
            print(f"Solved {sum(solved)}/{num_cubes} cubes")
            for i in range(num_cubes):
                status = "solved" if solved[i] else "unsolved"
                print(f"  cube {i}: {status} in {step_counts[i]} steps")
            pyglet.app.exit()
            return

        for i, cube in enumerate(cubes):
            if finished[i]:
                continue
            is_solved = solve_step(cube, model)
            step_counts[i] += 1
            if is_solved:
                solved[i] = True
                finished[i] = True
            elif step_counts[i] >= max_steps:
                finished[i] = True

    pyglet.clock.schedule_interval(step, 1 / 60)
    pyglet.app.run()


if __name__ == "__main__":
    test(render_indices=range(8))
