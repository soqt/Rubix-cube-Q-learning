import Cube

ACTIONS = ['F', 'R', 'B', 'L', 'U', 'D',
           'F2', 'R2', 'B2', 'L2', 'U2', 'D2',
           'F180','R180','B180','L180','U180','D180']


cube = Cube.Cube()
cube.random_cube_generator(20)
INITIAL_STATE = cube
print(INITIAL_STATE)

class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)

# Front90 = Operator("Move Front side (Green) 90 degree clockwise if Possible",\
#                    lambda s:can_move(s, ))

def can_move(s, side, degree):
    # if S is done. S.F, B, L U D is same color
    if s == "done":
        return False
    return True

def move(s, side, degree):
    pass

