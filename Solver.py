'''Solve scrambled cubes with a trained Q-network.

    python Solver.py --checkpoint checkpoints/latest.pt --scramble 20 --cubes 5

Uses batch weighted Q* search (Agostinelli et al., "A* Search Without Expansions: Learning
Heuristic Functions with Deep Q-Networks", arXiv:2102.04518): each iteration pops the --batch-size nodes with the lowest f = weight * g + h, and a
single network pass over a node scores all of its children, so children are only built once
popped. Lower --weight is greedier and faster; weight 1 gives shorter solutions but searches more.
'''
import argparse
import heapq
import itertools
import random
import time

import numpy as np
import torch

import Cube
import DeepQ


def standard_notation(moves):
    '''This repo's move names ('F', 'F2' = counter-clockwise, 'F180') in standard notation (F, F', F2).'''
    return ' '.join(move[0] + {1: '', 2: "'", 4: '2'}[len(move)] for move in moves)


def random_scramble(n):
    '''n random moves, never turning the same face twice in a row.'''
    moves = []
    while len(moves) < n:
        move = random.choice(Cube.POSSIBLE_MOVES)
        if not moves or move[0] != moves[-1][0]:
            moves.append(move)
    return moves


class QFunction:
    '''Runs the network on numpy states and returns numpy Q-values.'''

    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, states):
        return self.model(torch.from_numpy(states).to(self.device)).cpu().numpy()


def q_star_search(q_function, start, weight=0.6, batch_size=1000, max_nodes=5_000_000, time_limit=None):
    '''Returns (list of move names or None if no solution was found within the limits, nodes generated).'''
    goal = Cube.SOLVED_STATE.tobytes()
    start = np.asarray(start, dtype=np.uint8)
    if start.tobytes() == goal:
        return [], 1
    face = np.arange(DeepQ.N_ACTIONS) % 6

    states = np.empty((1 << 16, 54), np.uint8)
    states[0] = start
    parent, action, cost = [-1], [-1], [0]
    best_cost = {start.tobytes(): 0}
    open_list, tiebreak = [], itertools.count()
    new_nodes = [0]
    started = time.time()

    while True:
        # Q(s, a) - 1 estimates the moves left after reaching child a(s); turning the face that
        # was just turned again is never useful, since every combination is already one move
        nodes = np.array(new_nodes)
        g = np.array([cost[n] for n in new_nodes])
        last = np.array([action[n] for n in new_nodes])
        f = weight * (g[:, None] + 1) + q_function(states[nodes]) - 1
        f[(last[:, None] >= 0) & (face[None, :] == face[last][:, None])] = np.inf
        for node, row in zip(new_nodes, f.tolist()):
            for a, priority in enumerate(row):
                if priority != np.inf:
                    heapq.heappush(open_list, (priority, next(tiebreak), node, a))

        new_nodes = []
        while not new_nodes:
            if not open_list or len(parent) >= max_nodes or (time_limit and time.time() - started > time_limit):
                return None, len(parent)
            batch = [heapq.heappop(open_list) for _ in range(min(batch_size, len(open_list)))]
            parents = np.array([item[2] for item in batch])
            actions = np.array([item[3] for item in batch])
            children = np.take_along_axis(states[parents], Cube.MOVE_PERMS[actions], axis=1)
            for p, a, child in zip(parents.tolist(), actions.tolist(), children):
                key, g_child = child.tobytes(), cost[p] + 1
                if best_cost.get(key, g_child + 1) <= g_child:
                    continue
                best_cost[key] = g_child
                node = len(parent)
                if node == len(states):
                    states = np.concatenate([states, np.empty_like(states)])
                states[node] = child
                parent.append(p)
                action.append(a)
                cost.append(g_child)
                if key == goal:
                    moves = []
                    while action[node] >= 0:
                        moves.append(Cube.POSSIBLE_MOVES[action[node]])
                        node = parent[node]
                    return moves[::-1], len(parent)
                new_nodes.append(node)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='checkpoints/latest.pt')
    p.add_argument('--scramble', type=int, default=20, help='random moves per scramble')
    p.add_argument('--cubes', type=int, default=5, help='number of cubes to scramble and solve')
    p.add_argument('--weight', type=float, default=0.6, help='weight on the path cost g in f = weight * g + h')
    p.add_argument('--batch-size', type=int, default=1000, help='nodes expanded per network pass')
    p.add_argument('--max-nodes', type=int, default=5_000_000)
    p.add_argument('--time-limit', type=float, default=600, help='seconds per cube')
    p.add_argument('--device', help='cuda, mps or cpu (default: best available)')
    p.add_argument('--seed', type=int)
    args = p.parse_args()

    random.seed(args.seed)
    q_function = QFunction(DeepQ.load_model(args.checkpoint, DeepQ.select_device(args.device)))
    solved, lengths = 0, []
    for i in range(args.cubes):
        scramble = random_scramble(args.scramble)
        cube = Cube.Cube()
        for move in scramble:
            cube.turn(move)
        started = time.time()
        moves, nodes = q_star_search(q_function, cube.to_array(), args.weight, args.batch_size,
                                     args.max_nodes, args.time_limit)
        elapsed = time.time() - started
        print(f'cube {i + 1}: scramble {standard_notation(scramble)}')
        if moves is None:
            print(f'  not solved ({nodes} nodes, {elapsed:.1f}s)')
            continue
        for move in moves:
            cube.turn(move)
        assert cube.is_solved()
        solved += 1
        lengths.append(len(moves))
        print(f'  solved in {len(moves)} moves ({nodes} nodes, {elapsed:.1f}s): {standard_notation(moves)}')
    if lengths:
        print(f'solved {solved}/{args.cubes}, mean solution length {np.mean(lengths):.1f}')
    else:
        print(f'solved 0/{args.cubes}')


if __name__ == '__main__':
    main()
