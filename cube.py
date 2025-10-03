import os
import random
from cubePieces import Piece
from Constants import colors
from itertools import product

FRONT_COLORS, TOP_COLORS, SIDE_COLORS = colors

class Cube:
    def __init__(self):
        self._pieces: list[list[Piece]] = []
        self.make_solved_cube()
    
    def __repr__(self):
        return f"\n{repr(self._pieces[0][0:3])}\n{repr(self._pieces[0][3:6])}\n{repr(self._pieces[0][6:9])}" +\
               f"\n{repr(self._pieces[1][0:3])}\n{repr(self._pieces[1][3:6])}\n{repr(self._pieces[1][6:9])}" +\
               f"\n{repr(self._pieces[2][0:3])}\n{repr(self._pieces[2][3:6])}\n{repr(self._pieces[2][6:9])}"

    def __eq__(self, other):
        if not isinstance(other, Cube):
            return NotImplemented
        return self._pieces == other._pieces

    def get_state(self):
        state = []
        for face in self._pieces:
            for piece in face:
                state.extend(piece.get_colors())
        return state

    def is_solved(self):
        face_size = 6
        for face in self._pieces:
            colors = [piece.get_colors()[0] for piece in face]
            if len(set(colors)) != 1:
                return False
        return True

    def make_solved_cube(self):
        corners: list[Piece] = []
        sides: list[Piece] = []
        centers: list[Piece] = []

        for i,j,k in product(FRONT_COLORS,TOP_COLORS, SIDE_COLORS):
            corners.append(Piece([i,j,k]))
            if Piece([j,i]) not in sides and Piece([i,j]) not in sides:
                sides.append(Piece([i,j]))
            if Piece([k,j]) not in sides and Piece([j,k]) not in sides:
                sides.append(Piece([j,k]))
            if Piece([k,i]) not in sides and Piece([i,k]) not in sides:
                sides.append(Piece([i,k]))

        centers = [(Piece(color)) for color in FRONT_COLORS + SIDE_COLORS + TOP_COLORS]
        corners.sort(reverse=True)
        sides.sort(reverse=True)
        centers.sort(reverse=True)
        self._pieces = \
        [[corners[0].reverse_y_rotation(), sides[0].reverse_y_rotation(), corners[1].reverse_y_rotation().x_flip(), 
        sides[2].x_flip(), centers[0], sides[3],
        corners[2].z_flip().y_rotation(), sides[1].y_rotation(), corners[3].y_rotation()],
        [sides[8].z_rotation().x_flip(), centers[2].z_rotation(), sides[9].z_rotation(),
        centers[4].reverse_x_rotation(), Piece('X'), centers[5].x_rotation(),
        sides[10].x_flip().reverse_z_rotation(), centers[3].reverse_z_rotation(), sides[11].reverse_z_rotation()],
        [corners[4].reverse_y_rotation().y_flip(), sides[4].reverse_y_rotation().y_flip(), corners[5].y_flip().reverse_y_rotation().x_flip(),
        sides[6].x_rotation().x_rotation(), centers[1].y_flip(), sides[7].y_flip(),
        corners[6].y_flip().x_flip().reverse_y_rotation(), sides[5].y_rotation().y_flip(), corners[7].y_rotation().y_flip()]]
        
    def U(self):
        indexes = [(i,j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.x_rotation, [
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                          ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def Ud(self):
        indexes = [(i,j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.reverse_x_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def D(self):
        indexes = [(i,-1-j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.x_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def Dd(self):
        indexes = [(i,-1-j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.reverse_x_rotation,[
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def R(self):
        indexes = [(i,-1-3*j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.z_rotation, [
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def Rd(self):
        indexes = [(i,-1-3*j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.reverse_z_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def L(self):
        indexes = [(i, 3*j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.reverse_z_rotation, [
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def Ld(self):
        indexes = [(i, 3*j) for i in range(3) for j in range(3)]
        face = [self._pieces[i][j] for (i,j) in indexes]
        face = list(map(Piece.z_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))
        for (i,j), piece in zip(indexes, face):
            self._pieces[i][j] = piece

    def F(self):
        face = self._pieces[0]
        self._pieces[0] = list(map(Piece.y_rotation, [
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                        ]))
    
    def Fd(self):
        face = self._pieces[0]
        self._pieces[0] = list(map(Piece.reverse_y_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))

    def B(self):
        face = self._pieces[2]
        self._pieces[2] = list(map(Piece.reverse_y_rotation, [
                        face[2], face[5], face[8],
                        face[1], face[4], face[7],
                        face[0], face[3], face[6]
                        ]))

    def Bd(self):
        face = self._pieces[2]
        self._pieces[2] = list(map(Piece.y_rotation, [
                        face[6], face[3], face[0],
                        face[7], face[4], face[1],
                        face[8], face[5], face[2]
                        ]))

    @property
    def moves(self):
        return ["U","Ud","D","Dd","R","Rd","L","Ld","F","Fd","B","Bd"]    

    def scramble(self):
        for _ in range(25):
            move = random.choice(self.moves)
            getattr(self, move)()