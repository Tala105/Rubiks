from functools import total_ordering
from Constants import colors

FRONT_COLORS, TOP_COLORS, SIDE_COLORS = colors
color_onehotencoding = {
    'W': 0,
    'Y': 1,
    'R': 2,
    'O': 3,
    'G': 4,
    'B': 5
}

def compare_colors(color1: str, color2: str, index: int) -> bool:
    colors_list = colors[index]
    
    if color1 in colors_list:
        if color2 not in colors_list:
            return True
        if colors_list.index(color1) < colors_list.index(color2):
            return True
        
    if color2 in colors_list:
        return False
    if index == 2:
        return False

    return compare_colors(color1, color2, index + 1)

@total_ordering
class Piece:
    def __init__(self, colors: tuple[str, ...]):
        if len(colors) == 3:
            self.colors = {'F': colors[0], 'B': 'X', 'R': colors[1], 'L': 'X', 'T': colors[2], 'D': 'X'}
            self.type = 'corner'
        if len(colors) == 2:
            self.colors = {'F': colors[0], 'B': 'X', 'R': colors[1], 'L': 'X', 'T': 'X', 'D': 'X'}
            self.type = 'side'
        if len(colors) == 1:
            self.colors = {'F': colors[0], 'B': 'X', 'R': 'X', 'L': 'X', 'T': 'X', 'D': 'X'}
            self.type = 'center'
    
    def __repr__(self):
        return f"{tuple((face, color) for face, color in self.colors.items() if color != 'X')}"
    
    def __str__(self):
        if self.type == 'corner':
            return f"Piece Type: Corner, Front: {self.colors['F']}, Right: {self.colors['R']}, Top: {self.colors['T']}"
        if self.type == 'side':
            return f"Piece Type: Side, Front: {self.colors['F']}, Right: {self.colors['R']}"
        if self.type == 'center':
            return f"Piece Type: Center, Color: {self.colors['F']}"
        return "Invalid piece"

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return NotImplemented
        return self.colors == other.colors
    
    def __gt__(self, other):
        if not isinstance(other, Piece):
            return NotImplemented

        if compare_colors(self.colors['F'], other.colors['F'], 0):
            return True
        if self.colors['F'] == other.colors['F']:
            if compare_colors(self.colors['R'], other.colors['R'], 0):
                return True
            if self.colors['R'] == other.colors['R']:
                if compare_colors(self.colors['T'], other.colors['T'], 0):
                    return True
        return False

    def get_colors(self):
        output = [color_onehotencoding[color] for color in self.colors.values() if color != 'X']
        return output

    def x_rotation(self):
        self.colors['F'], self.colors['B'], self.colors['R'], self.colors['L'] = \
        self.colors['L'], self.colors['R'], self.colors['F'], self.colors['B']
        return self

    def y_rotation(self):
        self.colors['R'], self.colors['L'], self.colors['T'], self.colors['D'] = \
        self.colors['T'], self.colors['D'], self.colors['L'], self.colors['R']
        return self
    
    def z_rotation(self):
        self.colors['F'], self.colors['B'], self.colors['T'], self.colors['D'] = \
        self.colors['D'], self.colors['T'], self.colors['F'], self.colors['B']
        return self

    def reverse_x_rotation(self):
        self.colors['F'], self.colors['B'], self.colors['R'], self.colors['L'] = \
        self.colors['R'], self.colors['L'], self.colors['B'], self.colors['F']
        return self

    def reverse_y_rotation(self):
        self.colors['R'], self.colors['L'], self.colors['T'], self.colors['D'] = \
        self.colors['D'], self.colors['T'], self.colors['R'], self.colors['L']
        return self

    def reverse_z_rotation(self):
        self.colors['F'], self.colors['B'], self.colors['T'], self.colors['D'] = \
        self.colors['T'], self.colors['D'], self.colors['B'], self.colors['F']
        return self
    
    def x_flip(self):
        self.colors['R'], self.colors['L'] = self.colors['L'], self.colors['R']
        return self

    def y_flip(self):
        self.colors['F'], self.colors['B'] = self.colors['B'], self.colors['F']
        return self
    
    def z_flip(self):
        self.colors['T'], self.colors['D'] = self.colors['D'], self.colors['T']
        return self
