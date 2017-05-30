import RubixCube
import Cube
from collections import defaultdict
import pickle

CORNERS = {}
HALF_EDGES = {}
OTHER_HALP_EDGES= {}


def runBFS():
    initial_state = Cube.Cube()
    global COUNT, BACKLINKS
    COUNT = 0
    IterativeBFS(initial_state)
    print(str(COUNT) + " states examined.")
    return


def IterativeBFS(initial_state):
    global COUNT, CORNERS, HALF_EDGES, OTHER_HALP_EDGES

    OPEN = [initial_state]
    CLOSED = []
    step = 0
    while OPEN != []:
        S = OPEN[0]
        del OPEN[0]
        CLOSED.append(S)
        COUNT += 1

        L = []
        neigbors = [op.apply(S) for op in RubixCube.OPERATORS]
        for new_state in neigbors:
            if new_state.get_corner_state() not in CORNERS and \
                new_state.get_half_edge_state() not in HALF_EDGES and \
                new_state.get_other_half_edge_state() not in OTHER_HALP_EDGES:
                    L.append(new_state)
        OPEN = OPEN + L

        corners = S.get_corner_state()
        if corners not in CORNERS:
            CORNERS[corners] = step
        half_edges = S.get_half_edge_state()
        if half_edges not in HALF_EDGES:
            HALF_EDGES[half_edges] = step
        other_half_edges = S.get_other_half_edge_state()
        if other_half_edges not in OTHER_HALP_EDGES:
            OTHER_HALP_EDGES[other_half_edges] = step

        step += 1

def save_obj(obj, name):
    with open('pattern_database/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('pattern_database/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

runBFS()
save_obj(CORNERS, "corners")
save_obj(HALF_EDGES, "half_edges")
save_obj(OTHER_HALP_EDGES, "other_half_edges")

test = load_obj("corners")
test2 = load_obj("half_edges")
test3 = load_obj("other_half_edges")
print(test)
print(test2)
print(test3)