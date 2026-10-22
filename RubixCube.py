import Cube
import copy

ACTIONS = Cube.POSSIBLE_MOVES


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


def describe(action):
    turn = {1: "90 degrees clockwise", 2: "90 degrees counter-clockwise", 4: "180 degrees"}[len(action)]
    return "Move " + action[0] + " side " + turn


OPERATORS = [Operator(describe(action),
                      lambda s, action=action: can_move(s, action),
                      lambda s, action=action: move(s, action))
             for action in ACTIONS]

def can_move(s, action):
    # the solved cube is terminal
    return h_manhattan(s) != 0

def move(s, action):
    state = copy.deepcopy(s)
    state.turn(action)
    return state

def T(s, a, sp):
    '''Turning a face is deterministic.'''
    return 1.0 if move(s, a) == sp else 0.0



def R_func(s, a, sp):
    '''Return the reward associated with transitioning from s to sp via action a.
    Every move costs 1 and the solved cube is terminal, so the best policy solves the cube in as
    few moves as possible. DeepQ.py learns the same values as costs: Q_cost(s, a) = -Q_reward(s, a).'''
    return -1.0



def h_manhattan(state):
    '''Number of stickers that don't match the centre of their face (0 when solved).'''
    return int(sum((face != face[1, 1]).sum() for face in state.faces.values()))

HEURISTICS = {'h_manhattan' : h_manhattan}
