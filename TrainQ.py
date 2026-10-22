'''Train the Rubik's cube Q-network with deep Q-learning.

    python TrainQ.py              # defaults are sized for a 16 GB GPU such as an RTX 5080
    python TrainQ.py --resume     # continue from the last checkpoint (Ctrl-C also saves one)

Every step scrambles a batch of cubes on the GPU with 1..max_scramble random moves and regresses
Q(s, a) towards the Bellman target 1 + min_a' Q_target(a(s), a'), using 0 for the min when a(s) is
solved. As in DeepCubeA, the target network is only replaced by the online network once the
online network has fit it (mean loss < --loss-thresh), so cost-to-go estimates spread outwards
from the solved cube one move at a time.
'''
import argparse
import contextlib
import copy
import os
import time

import torch
import torch.nn.functional as F

import DeepQ

EVAL_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=1_000_000, help='total training steps')
    p.add_argument('--batch-size', type=int, default=10_000, help='scrambled cubes per step')
    p.add_argument('--max-scramble', type=int, default=25, help='max random moves per training scramble')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-decay', type=float, default=0.9999993, help='learning rate multiplier per step')
    p.add_argument('--update-interval', type=int, default=1000,
                   help='steps between target-network checks, evaluations and checkpoints')
    p.add_argument('--loss-thresh', type=float, default=0.06,
                   help='replace the target network when the mean loss since the last check is below this')
    p.add_argument('--h1', type=int, default=5000, help='width of the first hidden layer')
    p.add_argument('--h2', type=int, default=1000, help='width of the residual layers')
    p.add_argument('--res-blocks', type=int, default=4)
    p.add_argument('--target-chunk', type=int, default=60_000,
                   help='max cubes per target-network forward pass; lower it if the GPU runs out of memory')
    p.add_argument('--eval-cubes', type=int, default=1000, help='cubes per scramble depth in the greedy evaluation')
    p.add_argument('--log-interval', type=int, default=100)
    p.add_argument('--checkpoint', default=os.path.join('checkpoints', 'latest.pt'))
    p.add_argument('--resume', action='store_true', help='continue training from --checkpoint')
    p.add_argument('--device', help='cuda, mps or cpu (default: best available)')
    p.add_argument('--no-amp', action='store_true', help='train in fp32 instead of bf16 autocast on CUDA')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the network (needs Triton; on Windows: pip install triton-windows)')
    p.add_argument('--seed', type=int)
    return p.parse_args()


def bellman_targets(target_net, env, states, chunk):
    children = env.children(states).flatten(0, 1)
    cost_to_go = torch.cat([target_net(c).min(1).values for c in children.split(chunk)])
    cost_to_go = cost_to_go.clamp_min(0).masked_fill(env.is_solved(children), 0)
    return (1 + cost_to_go).view(len(states), DeepQ.N_ACTIONS)


def evaluate(model, env, max_scramble, n):
    '''Fraction of cubes scrambled with exactly d moves that the greedy policy solves in 2d moves.'''
    model.eval()
    rates = {}
    for depth in [d for d in EVAL_DEPTHS if d <= max_scramble]:
        states = env.scramble(n, depth, exact=True)
        rates[depth] = DeepQ.greedy_solve(model, env, states, 2 * depth).float().mean().item()
    model.train()
    return rates


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def main():
    args = parse_args()
    device = DeepQ.select_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    env = DeepQ.TorchCube(device)

    checkpoint = torch.load(args.checkpoint, map_location=device) if args.resume else None
    config = checkpoint['config'] if checkpoint else {'h1': args.h1, 'h2': args.h2, 'res_blocks': args.res_blocks}
    model = DeepQ.QNetwork(**config).to(device)
    target = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    step = target_updates = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        target.load_state_dict(checkpoint['target'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step, target_updates = checkpoint['step'], checkpoint['target_updates']
        print(f'resumed {args.checkpoint} at step {step} ({target_updates} target updates)')
    target.eval().requires_grad_(False)

    train_net, target_net = (torch.compile(model), torch.compile(target)) if args.compile else (model, target)
    use_amp = device.type == 'cuda' and not args.no_amp
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    def checkpoint_state():
        return {'config': config, 'model': model.state_dict(), 'target': target.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'step': step, 'target_updates': target_updates}

    name = torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type
    params = sum(p.numel() for p in model.parameters())
    print(f'device {name} | {params / 1e6:.1f}M parameters | bf16 autocast {use_amp} | batch {args.batch_size} '
          f'| scrambles 1..{args.max_scramble}')

    model.train()
    interval_loss = torch.zeros((), device=device)
    interval_steps = 0
    log_loss = torch.zeros((), device=device)
    log_time = time.time()
    try:
        while step < args.steps:
            states = env.scramble(args.batch_size, args.max_scramble)
            with torch.no_grad(), autocast():
                targets = bellman_targets(target_net, env, states, args.target_chunk)
            with autocast():
                q = train_net(states)
            loss = F.mse_loss(q, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            interval_steps += 1
            interval_loss += loss.detach()
            log_loss += loss.detach()

            if step % args.log_interval == 0:
                now = time.time()
                print(f'step {step:>8} | loss {log_loss.item() / args.log_interval:7.4f} '
                      f'| lr {scheduler.get_last_lr()[0]:.2e} | {args.log_interval / (now - log_time):5.1f} steps/s '
                      f'| target updates {target_updates}', flush=True)
                log_loss.zero_()
                log_time = now

            if step % args.update_interval == 0:
                mean_loss = interval_loss.item() / interval_steps
                if mean_loss < args.loss_thresh:
                    target.load_state_dict(model.state_dict())
                    target_updates += 1
                rates = evaluate(model, env, args.max_scramble, args.eval_cubes)
                print(f'  check: mean loss {mean_loss:.4f} -> target network '
                      f'{"updated (#%d)" % target_updates if mean_loss < args.loss_thresh else "kept"}')
                print('  greedy solve % by scramble depth: ' +
                      '  '.join(f'{d}:{100 * r:.0f}' for d, r in rates.items()), flush=True)
                save_checkpoint(args.checkpoint, checkpoint_state())
                interval_loss.zero_()
                interval_steps = 0
                log_time = time.time()
    except KeyboardInterrupt:
        print('interrupted')
    save_checkpoint(args.checkpoint, checkpoint_state())
    print(f'saved {args.checkpoint} at step {step}')


if __name__ == '__main__':
    main()
