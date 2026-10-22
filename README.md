# Rubix-cube-Q-learning

Solves the Rubik's cube with deep Q-learning on the GPU. A network learns Q(s, a), the number of
moves needed to solve the cube after doing move a in state s, from cubes scrambled on the GPU, the
same way [DeepCubeA](https://www.nature.com/articles/s42256-019-0070-z) trains its cost-to-go
network. Scrambled cubes are then solved with batch weighted
[Q* search](https://arxiv.org/abs/2102.04518), which scores all 18 children of a node with one
network pass.

| File | |
|---|---|
| `Cube.py` | The cube. Moves are sticker permutations derived from 3D geometry (`MOVE_PERMS`) |
| `DeepQ.py` | Batched GPU cube simulator and the Q-network |
| `TrainQ.py` | Training |
| `Solver.py` | Q* search; scrambles and solves cubes from the command line |
| `RunApp.py` | tkinter GUI that scrambles a cube and animates the solution |
| `RubixCube.py`, `MDP.py` | MDP formulation: operators, transition, reward, heuristic |
| `Pattern_db_generator.py` | Corner and edge pattern databases (`pattern_database/`) |

Move names: `F` turns the front face 90° clockwise, `F2` 90° counter-clockwise and `F180` 180°.
`Solver.py` also prints solutions in standard notation (`F`, `F'`, `F2`).

## Setup (RTX 5080)

RTX 50-series GPUs (Blackwell, sm_120) need an NVIDIA driver from the R570 series or newer and a
PyTorch build for CUDA 12.8 or newer; the default builds of older PyTorch versions don't include
sm_120 kernels.

```
python -m venv .venv
.venv\Scripts\activate              # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_arch_list())"
```

The last line should print your GPU and a list containing `sm_120`. `TrainQ.py` checks this
too and stops with instructions if the installed PyTorch can't run on the GPU.

## Training

```
python TrainQ.py
```

The defaults are DeepCubeA's: a 14.5M-parameter network, 10,000 cubes per step, scrambles of
1–25 moves, bf16 autocast. Every 1000 steps it checks whether the online network has fit the
target network well enough to replace it, prints how often the greedy policy solves cubes at each
scramble depth, and saves `checkpoints/latest.pt`:

```
  check: mean loss 0.0485 -> target network updated (#8)
  greedy solve % by scramble depth: 1:100  2:100  3:100  4:94  5:77  6:47
```

Short scrambles get solved first and the solve rate climbs to deeper ones as training goes on.
DeepCubeA trained for 1M steps, so expect a run of many hours to over a day on one GPU for a
network that solves fully scrambled cubes. Ctrl-C saves a checkpoint, and `--resume` continues
from it.

* Out of GPU memory: lower `--target-chunk` (cubes per target-network pass) or `--batch-size`.
* Faster: `--compile` uses `torch.compile` (needs Triton; on Windows `pip install triton-windows`).
* Quick experiment: a smaller network and shorter scrambles, e.g.
  `python TrainQ.py --h1 1000 --h2 500 --res-blocks 2 --batch-size 2000 --max-scramble 7`
  learns to solve most 6-move scrambles in a few minutes.

## Solving

```
python Solver.py --scramble 20 --cubes 5
python RunApp.py --scramble 20 --checkpoint checkpoints/latest.pt
```

`--weight` trades solution length for speed (lower is greedier), `--batch-size` sets how many
nodes are expanded per network pass, and `--time-limit` / `--max-nodes` bound the search.

## How it works

The MDP has a reward of -1 per move and the solved cube is terminal, so the best policy solves the
cube in as few moves as possible. Written as a cost, the Bellman equation is

    Q(s, a) = 1 + min_a' Q(a(s), a')       (and 1 if a(s) is solved)

Each training step scrambles a batch of cubes, builds all 18 children of each, evaluates the
target network on the children and regresses the online network's 18 outputs towards these
targets. The target network is only replaced once the online network has fit it
(`--loss-thresh`), so correct values spread outwards from the solved cube one move at a time.
