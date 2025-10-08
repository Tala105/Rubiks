import os
import pyglet
from pyglet.gl import *
from pyglet.graphics import Batch
from pyglet.graphics.shader import Shader, ShaderProgram, ShaderSource
from pyglet.window import key, mouse, Window
from cube import Cube
from cubePieces import Piece
from pyglet.math import Mat4, Vec3, Vec4
from math import radians

os.system('clear')

color_map = {
    'W': (1, 1, 1, 1),
    'Y': (1, 1, 0, 1),
    'R': (1, 0, 0, 1),
    'O': (1, 0.5, 0, 1),
    'G': (0, 1, 0, 1),
    'B': (0, 0, 1, 1),
    'X': (0.2, 0.2, 0.2, 1.0)
}

cube_moves: dict[int, tuple[str, str]]= {
    key.W: ('U', 'Ud'),
    key.S: ('D', 'Dd'),
    key.A: ('L', 'Ld'),
    key.D: ('R', 'Rd'),
    key.E: ('F', 'Fd'),
    key.Q: ('B', 'Bd')
}

vertex_source = """#version 460
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

fragment_source = """#version 460
    in vec4 vertex_colors;
    out vec4 final_color;

    void main()
    {
        if (vertex_colors.a == 0.0) discard;
        final_color = vertex_colors;
    }
"""

vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
program = ShaderProgram(vert_shader, frag_shader)


class WindowClass(Window):
    def __init__(self, pos: tuple[int,int] = (0,0), size: tuple[int,int] = (480,360)):
        super().__init__(*size, resizable=False, style=pyglet.window.Window.WINDOW_STYLE_OVERLAY) 
        self.set_location(*pos)
        self.cube: Cube = Cube()
        self.cube.scramble()
        self.rot_x = 0
        self.rot_y = 0
        self.mouse_down = False


    def on_draw(self):
        self.clear()
        self.batch = Batch()
        glEnable(GL_DEPTH_TEST)
        x_rot = radians(35.26)
        y_rot = radians(-45)
        self.projection = Mat4.orthogonal_projection(-3,3,-3,3,-255,255)
        self.view = Mat4.from_translation(Vec3(-2,2,-20))@Mat4.from_rotation(x_rot, Vec3(1,0,0))@\
            Mat4.from_rotation(y_rot, Vec3(0,1,0))
        pieces = sum(self.cube._pieces, [])
        size = 0.25
        for i, piece in enumerate(pieces):
            x_off = (i % 3) * 2 * size
            y_off = -(i // 3 - 3 * (i // 9)) * 2 * size
            z_off = -(i // 9) * 2 * size
            offset = (x_off, y_off, z_off)
            self.draw_piece(piece, offset, size=size)
        self.batch.draw()

    def on_resize(self, width, height):
        pass
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.RIGHT:
            pass
        pass

    def on_key_press(self, symbol, modifiers):
        if symbol in cube_moves.keys():
            if modifiers & key.MOD_SHIFT:
                 getattr(self.cube, cube_moves[symbol][1])()
            else:
                 getattr(self.cube, cube_moves[symbol][0])()
                 print('Moved')
        elif symbol == key.ENTER:
            self.cube.make_solved_cube()
        elif symbol == key.ESCAPE:
            self.cube.scramble()

    def draw_piece(self, piece: Piece, offset = (0,0,0), size = 1.0):
        s = size
        vertices = (
            [s, s, s], [s, -s, s], [-s, -s, s], [-s, s, s],
            [s, s, -s], [s, -s, -s], [-s, -s, -s], [-s, s, -s]
        )
        faces_index = (
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

        for i, f in enumerate(faces_index):
            positions = []
            for index in f:
                vertex = [x + y for x,y in zip(vertices[index], list(offset))]
                positions.extend(vertex)
            position = tuple(positions)
            v_list = program.vertex_list_indexed(4, pyglet.gl.GL_TRIANGLES,
                                        [0,1,2,0,2,3],
                                        position=('f', positions),
                                        colors=('f', face_colors[i]*4),
                                        batch=self.batch)

if __name__ == "__main__":
    n_cubes = 12
    for i in range(n_cubes):
        width = 1920//4
        height = 1080//3
        x = width*(i%4)
        y = height*(i//4)
        WindowClass((x, y), (width, height))

    pyglet.app.run()
