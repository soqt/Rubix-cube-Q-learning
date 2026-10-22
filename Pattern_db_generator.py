'''Pattern databases: for every pattern of Cube.get_corner_state(), get_half_edge_state() and
get_other_half_edge_state(), the fewest moves that produce it from the solved cube. That is a lower
bound on the moves needed to solve any cube showing the pattern; patterns missing from a database
need more than --max-depth moves.

    python Pattern_db_generator.py --max-depth 5

The search runs over patterns rather than whole cubes. The cube reached by the moves m1..mk is
SOLVED_STATE[p] with p = P[m1][P[m2]]...[P[mk]] (P = Cube.MOVE_PERMS), so its pattern is
SOLVED_STATE[p[stickers]]. Building p from the m1 end, p' = P[m][p], only ever needs p[stickers],
which keeps the search exact even for the edge patterns, whose slots also receive stickers from
edges outside the pattern.
'''
import argparse
import pickle

import numpy as np

import Cube

PATTERNS = {
    "corners": Cube.CORNER_STICKERS,
    "half_edges": Cube.HALF_EDGE_STICKERS,
    "other_half_edges": Cube.OTHER_HALF_EDGE_STICKERS,
}


def encode(patterns):
    '''Pattern rows (base-6 digits) as int64 keys; 24 digits still fit.'''
    keys = np.zeros(len(patterns), dtype=np.int64)
    for column in patterns.T:
        keys = keys * 6 + column
    return keys


def to_strings(patterns):
    digits = np.ascontiguousarray(patterns + ord('0'), dtype=np.uint8)
    return [s.decode() for s in digits.view(f'S{patterns.shape[1]}').ravel()]


def runBFS(stickers, max_depth):
    frontier = stickers[None, :].astype(np.uint8)
    seen = encode(Cube.SOLVED_STATE[frontier])
    database = dict.fromkeys(to_strings(Cube.SOLVED_STATE[frontier]), 0)
    for depth in range(1, max_depth + 1):
        children = Cube.MOVE_PERMS[:, frontier].reshape(-1, len(stickers)).astype(np.uint8)
        keys, first = np.unique(encode(Cube.SOLVED_STATE[children]), return_index=True)
        new = ~np.isin(keys, seen, assume_unique=True)
        if not new.any():
            break
        frontier = children[first[new]]
        seen = np.union1d(seen, keys[new])
        database.update(dict.fromkeys(to_strings(Cube.SOLVED_STATE[frontier]), depth))
        print(f"  depth {depth}: {len(frontier)} new patterns")
    return database


def save_obj(obj, name):
    with open('pattern_database/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('pattern_database/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-depth', type=int, default=4,
                        help='search depth; every extra level is roughly 13x bigger')
    args = parser.parse_args()
    for name, stickers in PATTERNS.items():
        print(name)
        database = runBFS(stickers, args.max_depth)
        save_obj(database, name)
        print(f"  {len(database)} patterns saved to pattern_database/{name}.pkl")
