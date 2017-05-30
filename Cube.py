import numpy as np
import copy
import random


COLOR = {
        0: "orange",
        1: "green",
        2: "red",
        3: "blue",
        4: "white",
        5: "yellow"
}

POSSIBLE_MOVES = ['F', 'R', 'B', 'L', 'U', 'D',
           'F2', 'R2', 'B2', 'L2', 'U2', 'D2',
           'F180','R180','B180','L180','U180','D180']

class Cube:
    def __init__(self, size = 3):
        self.size = 3
        self.faces = {
            "F": np.full((size, size), 0, dtype=int),
            "R": np.full((size, size), 1, dtype=int),
            "B": np.full((size, size), 2, dtype=int),
            "L": np.full((size, size), 3, dtype=int),
            "U": np.full((size, size), 4, dtype=int),
            "D": np.full((size, size), 5, dtype=int),
        }
    def __str__(self):
        output = ""
        for i in range(self.size):
            output += '       '+ str(self.faces["U"][i]) + '\n'
        for i in range(self.size):
            output += str(self.faces["L"][i]) + str(self.faces["F"][i]) + \
                      str(self.faces["R"][i]) + str(self.faces["B"][i]) + '\n'
        for i in range(self.size):
            output += '       '+ str(self.faces["D"][i]) + '\n'
        return output

    def turn_90(self, face):
        side = self.faces[face]
        self.faces[face] = np.rot90(side, 3)
        attaches = self.get_neighbors(face)
        temp = copy.deepcopy(attaches[0])
        self.update_row(attaches[3], attaches[0])
        self.update_row(attaches[2], attaches[3])
        self.update_row(attaches[1], attaches[2])
        self.update_row(temp, attaches[1])

    def turn_90_counter_clockwise(self, face):
        side = self.faces[face]
        self.faces[face] = np.rot90(side, 1)
        neighbors = self.get_neighbors(face)
        temp = copy.deepcopy(neighbors[3])
        self.update_row(neighbors[0], neighbors[3])
        self.update_row(neighbors[1], neighbors[0])
        self.update_row(neighbors[2], neighbors[1])
        self.update_row(temp, neighbors[2])

    def turn_180(self, face):
        side = self.faces[face]
        self.faces[face] = np.rot90(side, 1)
        neighbors = self.get_neighbors(face)
        temp = copy.deepcopy(neighbors[2])
        self.update_row(neighbors[0], neighbors[2])
        self.update_row(temp, neighbors[0])
        temp = copy.deepcopy(neighbors[3])
        self.update_row(neighbors[1], neighbors[3])
        self.update_row(temp, neighbors[1])

    def update_row(self, src, to):
        for i in range(3):
            to[i] = src[i]

    def get_neighbors(self, side):
        up = []
        left = []
        bottom = []
        right = []
        if side == 'F':
            left = self.faces['L'][:, self.size - 1]
            up = self.faces['U'][self.size-1]
            bottom = self.faces['D'][0]
            right = self.faces['R'][:, 0]
        elif side == 'L':
            left = self.faces['B'][:, self.size - 1]
            up = self.faces['U'][:, 0]
            bottom = self.faces['D'][:, 0]
            right = self.faces['F'][:, 0]
        elif side == 'B':
            left = self.faces['R'][:, self.size - 1]
            up = self.faces['U'][0]
            bottom = self.faces['D'][self.size -1]
            right = self.faces['L'][:, 0]
        elif side == 'R':
            left = self.faces['F'][:, self.size - 1]
            up = self.faces['U'][:, self.size - 1]
            bottom = self.faces['D'][:, self.size-1]
            right = self.faces['B'][:, 0]
        elif side == 'U':
            left = self.faces['L'][0]
            up = self.faces['B'][0]
            bottom = self.faces['F'][0]
            right = self.faces['R'][0]
        elif side == 'D':
            left = self.faces['L'][self.size - 1]
            up = self.faces['F'][self.size - 1]
            bottom = self.faces['B'][self.size - 1]
            right = self.faces['R'][self.size - 1]

        return (left, up, right, bottom)

    def generate_random_cube(self, steps=10):
        for i in range(steps):
            action = random.choice(POSSIBLE_MOVES)
            if len(action) == 1:
                self.turn_90(action)
            elif len(action) == 2:
                self.turn_90_counter_clockwise(action[0])
            else: self.turn_180(action[0])

    def turn(self, action):
        if len(action) == 1:
            self.turn_90(action)
        elif len(action) == 2:
            self.turn_90_counter_clockwise(action[0])
        else:
            self.turn_180(action[0])

    def get_corner_state(self):
        state = ""
        state += str(self.faces["U"].item(0)) + str(self.faces["L"].item(0)) + str(self.faces["B"].item(2))
        state += str(self.faces["F"].item(0)) + str(self.faces["L"].item(2)) + str(self.faces["U"].item(6))
        state += str(self.faces["F"].item(2)) + str(self.faces["U"].item(8)) + str(self.faces["R"].item(0))
        state += str(self.faces["F"].item(6)) + str(self.faces["L"].item(8)) + str(self.faces["D"].item(0))
        state += str(self.faces["D"].item(2)) + str(self.faces["F"].item(8)) + str(self.faces["R"].item(6))
        state += str(self.faces["L"].item(6)) + str(self.faces["D"].item(6)) + str(self.faces["B"].item(8))
        state += str(self.faces["R"].item(8)) + str(self.faces["D"].item(8)) + str(self.faces["B"].item(6))
        state += str(self.faces["U"].item(2)) + str(self.faces["R"].item(2)) + str(self.faces["B"].item(0))
        return state

    def get_half_edge_state(self):
        state = ""
        state += str(self.faces["F"].item(1)) + str(self.faces["U"].item(7)) + \
                 str(self.faces["F"].item(3)) + str(self.faces["L"].item(5)) + \
                 str(self.faces["F"].item(7)) + str(self.faces["D"].item(1)) + \
                 str(self.faces["F"].item(5)) + str(self.faces["R"].item(3)) + \
                 str(self.faces["L"].item(1)) + str(self.faces["U"].item(3)) + \
                 str(self.faces["U"].item(5)) + str(self.faces["R"].item(1))

        return state
    def get_other_half_edge_state(self):
        state = ""
        state += str(self.faces["B"].item(1)) + str(self.faces["U"].item(1)) + \
                 str(self.faces["B"].item(3)) + str(self.faces["R"].item(5)) + \
                 str(self.faces["B"].item(7)) + str(self.faces["D"].item(7)) + \
                 str(self.faces["B"].item(5)) + str(self.faces["L"].item(3)) + \
                 str(self.faces["D"].item(3)) + str(self.faces["L"].item(7)) + \
                 str(self.faces["D"].item(5)) + str(self.faces["R"].item(7))
        return state


# cube = Cube()
# hi = cube.get_corner_state()