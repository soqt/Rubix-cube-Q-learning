import numpy as np
import random


COLOR = {
        0: "orange",
        1: "green",
        2: "red",
        3: "blue",
        4: "white",
        5: "yellow"
}

# 'X' turns face X 90 degrees clockwise (looking straight at that face), 'X2' turns it
# 90 degrees counter-clockwise and 'X180' turns it 180 degrees.
POSSIBLE_MOVES = ['F', 'R', 'B', 'L', 'U', 'D',
           'F2', 'R2', 'B2', 'L2', 'U2', 'D2',
           'F180','R180','B180','L180','U180','D180']
MOVE_INDEX = {move: i for i, move in enumerate(POSSIBLE_MOVES)}

# Order of the faces in Cube.to_array(). Face i starts out (and its centre always stays) color i.
FACES = ['F', 'R', 'B', 'L', 'U', 'D']
SOLVED_STATE = np.repeat(np.arange(6, dtype=np.uint8), 9)


def _sticker(face, row, col):
    '''3D cubie position and outward normal of a sticker (x right, y up, z towards the viewer).
    Every face is stored as seen from outside, laid out like the usual net: U above F, D below F
    and L F R B from left to right.'''
    a, b = col - 1, 1 - row
    return {
        'F': ((a, b, 1), (0, 0, 1)),
        'B': ((-a, b, -1), (0, 0, -1)),
        'R': ((1, b, -a), (1, 0, 0)),
        'L': ((-1, b, a), (-1, 0, 0)),
        'U': ((a, 1, -b), (0, 1, 0)),
        'D': ((a, -1, b), (0, -1, 0)),
    }[face]


def _build_move_perms():
    '''MOVE_PERMS[m] is a permutation of the 54 stickers: after move m, state[i] = old_state[perm[i]].'''
    stickers = [_sticker(face, r, c) for face in FACES for r in range(3) for c in range(3)]
    index = {s: i for i, s in enumerate(stickers)}

    def rotate_clockwise(v, axis):
        # -90 degrees about the outward normal is clockwise when looking at the face
        return tuple(int(x) for x in np.dot(axis, v) * axis - np.cross(axis, v))

    clockwise = {}
    for f, face in enumerate(FACES):
        axis = np.array(stickers[9 * f + 4][1])
        perm = np.arange(54)
        for i, (pos, normal) in enumerate(stickers):
            if np.dot(pos, axis) == 1:
                perm[index[(rotate_clockwise(pos, axis), rotate_clockwise(normal, axis))]] = i
        clockwise[face] = perm

    perms = []
    for move in POSSIBLE_MOVES:
        quarter_turns = {1: 1, 2: 3, 4: 2}[len(move)]
        perm = np.arange(54)
        for _ in range(quarter_turns):
            perm = perm[clockwise[move[0]]]
        perms.append(perm)
    return np.stack(perms)


MOVE_PERMS = _build_move_perms()


def _sticker_indices(*stickers):
    return np.array([9 * FACES.index(face) + item for face, item in stickers])


CORNER_STICKERS = _sticker_indices(
    ('U', 0), ('L', 0), ('B', 2), ('F', 0), ('L', 2), ('U', 6), ('F', 2), ('U', 8),
    ('R', 0), ('F', 6), ('L', 8), ('D', 0), ('D', 2), ('F', 8), ('R', 6), ('L', 6),
    ('D', 6), ('B', 8), ('R', 8), ('D', 8), ('B', 6), ('U', 2), ('R', 2), ('B', 0))
HALF_EDGE_STICKERS = _sticker_indices(
    ('F', 1), ('U', 7), ('F', 3), ('L', 5), ('F', 7), ('D', 1),
    ('F', 5), ('R', 3), ('L', 1), ('U', 3), ('U', 5), ('R', 1))
OTHER_HALF_EDGE_STICKERS = _sticker_indices(
    ('B', 1), ('U', 1), ('B', 3), ('R', 5), ('B', 7), ('D', 7),
    ('B', 5), ('L', 3), ('D', 3), ('L', 7), ('D', 5), ('R', 7))


class Cube:
    def __init__(self, size = 3, h_value = -1):
        if size != 3:
            raise ValueError("only the 3x3x3 cube is supported")
        self.h_value = h_value
        self.size = size
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
    def __eq__(self, other):
        return isinstance(other, Cube) and np.array_equal(self.to_array(), other.to_array())
    def __hash__(self):
        return hash(self.to_array().tobytes())
    def get_h_value(self):
        return self.h_value

    def to_array(self):
        '''The 54 sticker colors as a uint8 array: faces in FACES order, each row by row.'''
        return np.concatenate([self.faces[face].ravel() for face in FACES]).astype(np.uint8)

    @classmethod
    def from_array(cls, state):
        cube = cls()
        for i, face in enumerate(FACES):
            cube.faces[face] = np.asarray(state[9 * i: 9 * i + 9], dtype=int).reshape(3, 3)
        return cube

    def is_solved(self):
        return np.array_equal(self.to_array(), SOLVED_STATE)

    def turn_90(self, face):
        self.turn(face)

    def turn_90_counter_clockwise(self, face):
        self.turn(face + '2')

    def turn_180(self, face):
        self.turn(face + '180')

    def generate_random_cube(self, steps=10):
        '''Scramble with `steps` random moves and return them.'''
        moves = [random.choice(POSSIBLE_MOVES) for _ in range(steps)]
        for action in moves:
            self.turn(action)
        return moves

    def turn(self, action):
        state = self.to_array()[MOVE_PERMS[MOVE_INDEX[action]]]
        for i, face in enumerate(FACES):
            self.faces[face] = state[9 * i: 9 * i + 9].astype(int).reshape(3, 3)

    def get_corner_state(self):
        return ''.join(str(c) for c in self.to_array()[CORNER_STICKERS])

    def get_half_edge_state(self):
        return ''.join(str(c) for c in self.to_array()[HALF_EDGE_STICKERS])

    def get_other_half_edge_state(self):
        return ''.join(str(c) for c in self.to_array()[OTHER_HALF_EDGE_STICKERS])
