import os
import importlib
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
from cubePieces import Piece
import cube
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

def draw_piece(piece: Piece, size=1):
    s = size
    vertices = (
        (s, s, s), (s, -s, s), (-s, -s, s), (-s, s, s),
        (s, s, -s), (s, -s, -s), (-s, -s, -s), (-s, s, -s)
    )
    faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5)
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

cube_instance = [cube.Cube()]

class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.startswith(".\cube") and not event.src_path.endswith("Render.py"):
            while True:
                try:
                    importlib.reload(cube)
                    cube_instance[0] = cube.Cube()
                    break
                except Exception as e:
                    print(f"Reload error: {e}")

observer = Observer()
observer.schedule(ReloadHandler(), path='.', recursive=False)
observer.start()

pygame.init()
display = (1920, 1080)
os.environ["SDL_VIDEO_WINDOW_POS"] = "1920,0"
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME)
gl.glEnable(gl.GL_DEPTH_TEST)

rot_x, rot_y = 0, 0
mouse_down = False
shift_down = False
last_mouse_pos = (0, 0)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mouse_down = True
                last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                mouse_down = False
            elif event.type == pygame.MOUSEMOTION and mouse_down:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                rot_x += dy
                rot_y += dx
                last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_down = True
                if event.key == pygame.K_w:
                    if shift_down:
                        cube_instance[0].Ud()
                    else:
                        cube_instance[0].U()
                elif event.key == pygame.K_s:
                    if shift_down:
                        cube_instance[0].Dd()
                    else:
                        cube_instance[0].D()
                elif event.key == pygame.K_a:
                    if shift_down:
                        cube_instance[0].Ld()
                    else:
                        cube_instance[0].L()
                elif event.key == pygame.K_d:
                    if shift_down:
                        cube_instance[0].Rd()
                    else:
                        cube_instance[0].R()
                elif event.key == pygame.K_e:
                    if shift_down:
                        cube_instance[0].Fd()
                    else:
                        cube_instance[0].F()
                elif event.key == pygame.K_q:
                    if shift_down:
                        cube_instance[0].Bd()
                    else:
                        cube_instance[0].B()
            elif event.type == pygame.K_ESCAPE:
                cube_instance[0].scramble()
            elif event.type == pygame.K_KP_ENTER:
                cube_instance[0].make_solved_cube()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_down = False

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        glu.gluPerspective(45, (display[0] / display[1]), 1, 150.0)
        gl.glTranslatef(-2, 3, -15)
        gl.glRotatef(rot_x, 1, 0, 0)
        gl.glRotatef(rot_y, 0, 1, 0)

        current_cube = cube_instance[0]
        pieces = current_cube._pieces[0] + current_cube._pieces[1] + current_cube._pieces[2]
        for i, piece in enumerate(pieces):
            gl.glPushMatrix()
            gl.glTranslatef((i % 3) * 2, -(i // 3 - 3 * (i // 9)) * 2, -(i // 9) * 2)
            draw_piece(piece)
            gl.glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(2)
except KeyboardInterrupt:
    observer.stop()
observer.join()
