'''Deep Q-learning for the Rubik's cube: a batched GPU cube simulator and the Q-network.

Q(s, a) is the estimated number of moves needed to solve the cube after doing action a in state s,
so the greedy policy is argmin_a Q(s, a). This is Q-learning with a reward of -1 per move written
as a cost (Q_cost = -Q_reward), trained the way DeepCubeA trains its cost-to-go network.
'''
import torch
from torch import nn
import torch.nn.functional as F

import Cube

N_ACTIONS = len(Cube.POSSIBLE_MOVES)
N_COLORS = 6
# Centres never move, so the network only looks at the other 48 stickers.
NON_CENTER = [i for i in range(54) if i % 9 != 4]


def select_device(name=None):
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        check_cuda_arch(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def check_cuda_arch(device):
    '''RTX 50-series GPUs (Blackwell, sm_120) need a PyTorch build for CUDA 12.8 or newer. Older
    builds only fail later with "no kernel image is available for execution on the device".'''
    major, minor = torch.cuda.get_device_capability(device)
    capability = major * 10 + minor
    arch_list = torch.cuda.get_arch_list()

    def arch_number(arch):
        return int(''.join(ch for ch in arch.split('_')[1] if ch.isdigit()))

    if f'sm_{capability}' in arch_list or any(
            arch.startswith('compute_') and arch_number(arch) <= capability for arch in arch_list):
        return
    raise SystemExit(
        f'{torch.cuda.get_device_name(device)} (sm_{capability}) is not supported by this PyTorch '
        f'build (torch {torch.__version__}, CUDA {torch.version.cuda}, built for {" ".join(arch_list)}).\n'
        'RTX 50-series GPUs need a build for CUDA 12.8 or newer, e.g.\n'
        '    pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128\n'
        '(https://pytorch.org/get-started/locally/ lists the current CUDA builds)')


class TorchCube:
    '''Batched cube simulator. States are (n, 54) uint8 tensors laid out like Cube.to_array().'''

    def __init__(self, device):
        self.device = torch.device(device)
        self.perms = torch.as_tensor(Cube.MOVE_PERMS, dtype=torch.long, device=self.device)
        self.goal = torch.as_tensor(Cube.SOLVED_STATE, device=self.device)

    def apply(self, states, actions):
        return states.gather(1, self.perms[actions])

    def children(self, states):
        '''(n, N_ACTIONS, 54): the result of every action from every state.'''
        return states[:, self.perms]

    def is_solved(self, states):
        return (states == self.goal).all(-1)

    def scramble(self, n, max_depth, exact=False):
        '''n cubes scrambled with a uniformly random number of moves in [1, max_depth]
        (exactly max_depth if exact). Consecutive moves never turn the same face.'''
        if exact:
            depths = torch.full((n,), max_depth, device=self.device)
        else:
            depths = torch.randint(1, max_depth + 1, (n,), device=self.device)
        states = self.goal.expand(n, -1).clone()
        face = torch.randint(0, 6, (n,), device=self.device)
        for t in range(max_depth):
            if t:
                face = (face + torch.randint(1, 6, (n,), device=self.device)) % 6
            # POSSIBLE_MOVES is 6 clockwise, then 6 counter-clockwise, then 6 half turns
            actions = face + 6 * torch.randint(0, 3, (n,), device=self.device)
            states = torch.where((t < depths)[:, None], self.apply(states, actions), states)
        return states


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        y = F.relu(self.bn1(self.fc1(x)))
        return F.relu(x + self.bn2(self.fc2(y)))


class QNetwork(nn.Module):
    '''DeepCubeA's network with one output per action. Takes raw (n, 54) uint8 states.'''

    def __init__(self, h1=5000, h2=1000, res_blocks=4):
        super().__init__()
        self.register_buffer('non_center', torch.tensor(NON_CENTER), persistent=False)
        self.body = nn.Sequential(
            nn.Linear(len(NON_CENTER) * N_COLORS, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(),
            *[ResidualBlock(h2) for _ in range(res_blocks)])
        self.head = nn.Linear(h2, N_ACTIONS)

    def forward(self, states):
        x = F.one_hot(states[:, self.non_center].long(), N_COLORS).flatten(1).float()
        x = self.body(x)
        # keep the output layer in fp32 under bf16 autocast: Q-values reach ~20, where bf16
        # can't tell 19.9 from 20
        with torch.autocast(x.device.type, enabled=False):
            return self.head(x.float())


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = QNetwork(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()


@torch.no_grad()
def greedy_solve(model, env, states, max_steps):
    '''Follow argmin_a Q(s, a) from every state. Returns a mask of the states that got solved.'''
    done = env.is_solved(states)
    for _ in range(max_steps):
        if done.all():
            break
        actions = model(states).argmin(1)
        states = torch.where(done[:, None], states, env.apply(states, actions))
        done |= env.is_solved(states)
    return done
