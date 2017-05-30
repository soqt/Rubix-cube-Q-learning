import Cube
import copy

ACTIONS = ['F', 'R', 'B', 'L', 'U', 'D',
           'F2', 'R2', 'B2', 'L2', 'U2', 'D2',
           'F180','R180','B180','L180','U180','D180']


cube = Cube.Cube()
cube.generate_random_cube(20)
INITIAL_STATE = cube

class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)


F = Operator("Move F side 90 degree clockwise if possible", lambda s: can_move(s, "F"), lambda s: move(s, 'F'))
R = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "R"),\
             lambda s: move(s, 'R'))
B = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "B"),\
             lambda s: move(s, 'B'))
L = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "L"),\
             lambda s: move(s, 'L'))
U = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "U"),\
             lambda s: move(s, 'U'))
D = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "D"),\
             lambda s: move(s, 'D'))
F2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "F"),\
             lambda s: move(s, 'F'))
R2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "R2"),\
             lambda s: move(s, 'R'))
B2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "B2"),\
             lambda s: move(s, 'B2'))
L2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "L2"),\
             lambda s: move(s, 'L2'))
U2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "U2"),\
             lambda s: move(s, 'U2'))
D2 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "D2"),\
             lambda s: move(s, 'D2'))
F180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "F180"),\
             lambda s: move(s, 'F180'))
R180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "R180"),\
             lambda s: move(s, 'R180'))
B180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "B180"),\
             lambda s: move(s, 'B180'))
L180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "L180"),\
             lambda s: move(s, 'L180'))
U180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "U180"),\
             lambda s: move(s, 'U180'))
D180 = Operator("Move F side 90 degree clockwise if possible",\
             lambda s: can_move(s, "D180"),\
             lambda s: move(s, 'D180'))

OPERATORS = [F, R, B, L, U, D, F2, R2, B2, L2, U2, D2, F180, R180, B180, L180, U180, D180]

def can_move(s, action):
    # if S is done. S.F, B, L U D is same color
    # if h_manhattan(s) == 0: return False
    return True

def move(s, action):
    state = copy.deepcopy(s)
    state.turn(action)
    return state

P_normal = 0.8
P_noise = 0.1
def T(s, a, sp):
    if h_manhattan(s) == 0: return 0
    return P_normal



def R_func(s, a, sp):
    '''Return the reward associated with transitioning from s to sp via action a.'''
    value = h_manhattan(s)
    arrived_value = h_manhattan(sp)
    if value == 9 or value == 0 or arrived_value == 9:
        return 100.0
    if value <= 18 or arrived_value <= 18:
        return 50.0
    if value <= 27 or arrived_value <= 27:
        return 30.0
    if arrived_value > value:
        return -20.0
    return -5.0



def h_manhattan(state):
    sum = 0
    face_count = 0
    for face in state.faces:
        for block in face:
            if block != face_count:
                sum += 1
        face_count += 1
    return sum

HEURISTICS = {'h_manhattan' : h_manhattan}