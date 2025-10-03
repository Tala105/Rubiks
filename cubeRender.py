import os
import importlib
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
from cubePieces import Piece
from cube import Cube
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

color_map = {
    'W': (1, 1, 1),
    'Y': (1, 1, 0),
    'R': (1, 0, 0),
    'O': (1, 0.5, 0),
    'G': (0, 1, 0),
    'B': (0, 0, 1),
    'X': (0.2, 0.2, 0.2)
}

class CubeRenderer:
    def __init__(self, cube_instance: list[Cube], display=(1920, 1080), window_pos=(1920, 0)):
        self.cube_instance = cube_instance
        self.display = display
        self.rot_x = 0
        self.rot_y = 0
        self.mouse_down = False
        self.shift_down = False
        self.last_mouse_pos = (0, 0)

        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{window_pos[0]},{window_pos[1]}"
        pygame.init()
        pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self._setup_hot_reload()

    def _setup_hot_reload(self):
        class ReloadHandler(FileSystemEventHandler):
            def __init__(self, renderer):
                self.renderer = renderer

            def on_modified(self, event):
                if event.src_path.startswith(".\\cube") and not event.src_path.endswith("Render.py"):
                    while True:
                        try:
                            import cube
                            self.renderer.cube_instance[0] = cube.Cube()
                            break
                        except Exception as e:
                            print(f"Reload error: {e}")

        self.observer = Observer()
        self.observer.schedule(ReloadHandler(self), path='.', recursive=False)
        self.observer.start()

    def draw_piece(self, piece: Piece, size=1):
        s = size
        vertices = (
            (s, s, s), (s, -s, s), (-s, -s, s), (-s, s, s),
            (s, s, -s), (s, -s, -s), (-s, -s, -s), (-s, s, -s)
        )
        faces = (
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (0, 3, 7, 4), (1, 2, 6, 5)
        )
        face_colors = {
            0: color_map[piece.colors['F']],
            1: color_map[piece.colors['B']],
            2: color_map[piece.colors['R']],
            3: color_map[piece.colors['L']],
            4: color_map[piece.colors['T']],
            5: color_map[piece.colors['D']],
        }
        gl.glBegin(gl.GL_QUADS)
        for i, f in enumerate(faces):
            gl.glColor3fv(face_colors[i])
            for v in f:
                gl.glVertex3fv(vertices[v])
        gl.glEnd()

        gl.glColor3fv((0, 0, 0))
        gl.glLineWidth(5)
        gl.glBegin(gl.GL_LINES)
        for f in faces:
            for i in range(4):
                gl.glVertex3fv(vertices[f[i]])
                gl.glVertex3fv(vertices[f[(i + 1) % 4]])
        gl.glEnd()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self.mouse_down = True
                self.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.mouse_down = False
            elif event.type == pygame.MOUSEMOTION and self.mouse_down:
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]
                self.rot_x += dy
                self.rot_y += dx
                self.last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    self.shift_down = True
                self._handle_cube_keys(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    self.shift_down = False

    def _handle_cube_keys(self, key):
        cube_moves = {
            pygame.K_w: (self.cube_instance[0].U, self.cube_instance[0].Ud),
            pygame.K_s: (self.cube_instance[0].D, self.cube_instance[0].Dd),
            pygame.K_a: (self.cube_instance[0].L, self.cube_instance[0].Ld),
            pygame.K_d: (self.cube_instance[0].R, self.cube_instance[0].Rd),
            pygame.K_e: (self.cube_instance[0].F, self.cube_instance[0].Fd),
            pygame.K_q: (self.cube_instance[0].B, self.cube_instance[0].Bd)
        }
        if key in cube_moves:
            move = cube_moves[key][1] if self.shift_down else cube_moves[key][0]
            move()
        elif key == pygame.K_ESCAPE:
            self.cube_instance[0].scramble()
        elif key == pygame.K_KP_ENTER:
            self.cube_instance[0].make_solved_cube()

    def render(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        glu.gluPerspective(45, (self.display[0] / self.display[1]), 1, 150.0)
        gl.glTranslatef(-2, 3, -15)
        gl.glRotatef(35.264,1,0,0)
        gl.glRotatef(45,0,1,0)
        gl.glRotatef(self.rot_x, 1, 0, 0)
        gl.glRotatef(self.rot_y, 0, 1, 0)

        current_cube = self.cube_instance[0]
        pieces = sum(current_cube._pieces, [])
        for i, piece in enumerate(pieces):
            gl.glPushMatrix()
            gl.glTranslatef((i % 3) * 2, -(i // 3 - 3 * (i // 9)) * 2, -(i // 9) * 2)
            self.draw_piece(piece)
            gl.glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(2)

    def run(self):
        try:
            print("Running...")
            while True:
                self.handle_input()
                self.render()
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

if __name__ == "__main__":
    cube_instance = [Cube()]
    renderer = CubeRenderer(cube_instance)
    renderer.run()