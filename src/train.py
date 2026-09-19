'''Train :class:`~src.model.FogOfWarNet` to guess what white cannot see.

Run shape
---------

The stop condition is a wall clock, not an epoch count.  The corpus is ~2.2
billion positions and an H100 gets through several million a second, so "an
epoch" is neither a useful unit of progress nor a thing the job has time to
round off; ``--hours`` is what the job is actually bounded by and the learning
rate is annealed against it, so the schedule finishes wherever the throughput
ends up landing.

The run survives being interrupted.  ``last.pt`` is written on a timer and on
``SIGUSR1``/``SIGTERM`` - which is what SLURM sends before it takes the node
away, given the ``--signal`` in ``scripts/train_model.sbatch`` - and
``--resume auto`` picks it up, elapsed clock and all, so a requeued job carries
on rather than restarting.

What is logged, and where
-------------------------

Everything lands under ``logs/<run>/``::

    train.log        what happened, in the order it happened
    metrics.csv      one row per --log-every steps, for plotting
    val.csv          one row per validation pass, plus per-piece recall
    progress.json    the latest numbers only, overwritten - for watching a job
    config.json      every argument this run was started with
    env.json         host, GPU, versions, git commit
    checkpoints/     last.pt, best.pt

Reading the numbers
-------------------

``acc`` over all 64 squares is not the metric: white can see about 45 of them
and copying its own input is free.  The ones that mean anything are the
``hidden_*`` columns, and the one to compare them against is ``baseline``, the
accuracy of answering "empty" for every hidden square - which is right about 85%
of the time, because most of the fog is over empty board.  ``hidden_piece_acc``,
over the hidden squares that actually hold something, is the hard half of the
task and starts at zero.
'''

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src import preprocess as pp
from src.model import (
    CLASS_EMPTY,
    CLASS_NAMES,
    FogOfWarNet,
    ModelConfig,
    N_CLASSES,
    parameter_breakdown,
    parameter_count,
)


REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group('data')
    data.add_argument('--shards', type=Path, default=pp.PROCESSED_DIR,
                      help='directory of .pt.xz shards from src/data.py')
    data.add_argument('--max-shards', type=int, default=0,
                      help='use only the first N shards; 0 means all of them')
    data.add_argument('--val-shards', type=int, default=32,
                      help='whole shards held out of training for validation')
    data.add_argument('--split-seed', type=int, default=0,
                      help='which shards are held out; keep it fixed across runs')
    data.add_argument('--workers', type=int, default=24,
                      help='loader processes; decompressing shards is the CPU cost')
    data.add_argument('--pool-shards', type=int, default=4,
                      help='shards mixed together before a batch is drawn from them')
    data.add_argument('--no-cache', action='store_true',
                      help='read shards from disk every pass instead of holding them in RAM')
    data.add_argument('--cache-threads', type=int, default=16,
                      help='parallel reads while filling the RAM cache')

    # Defaults come from ModelConfig rather than being repeated here, so that
    # resizing the model is a one-line change in src/model.py.
    shape = ModelConfig()
    model = parser.add_argument_group('model')
    model.add_argument('--d-model', type=int, default=shape.d_model)
    model.add_argument('--n-heads', type=int, default=shape.n_heads)
    model.add_argument('--n-layers', type=int, default=shape.n_layers)
    model.add_argument('--d-ff', type=int, default=shape.d_ff)
    model.add_argument('--dropout', type=float, default=shape.dropout)

    optim = parser.add_argument_group('optimisation')
    optim.add_argument('--batch-size', type=int, default=16384,
                       help='positions per step; each one is 64 tokens')
    optim.add_argument('--lr', type=float, default=3e-3)
    optim.add_argument('--min-lr-frac', type=float, default=0.05,
                       help='fraction of --lr the cosine decays to')
    optim.add_argument('--warmup-steps', type=int, default=2000)
    optim.add_argument('--weight-decay', type=float, default=0.01)
    optim.add_argument('--beta1', type=float, default=0.9)
    optim.add_argument('--beta2', type=float, default=0.95)
    optim.add_argument('--grad-clip', type=float, default=1.0)
    optim.add_argument('--hidden-weight', type=float, default=1.0,
                       help='weight on hidden squares in the loss; visible ones are '
                            'always weighted 1 and are nearly free to get right')
    optim.add_argument('--schedule', choices=('time', 'step'), default='time',
                       help='anneal the learning rate against the wall clock or --max-steps')

    run = parser.add_argument_group('run')
    run.add_argument('--hours', type=float, default=11.5,
                     help='wall-clock budget; leave slack inside the SLURM limit')
    run.add_argument('--max-steps', type=int, default=0,
                     help='stop after N steps as well; 0 means only --hours stops it')
    run.add_argument('--name', default='fow',
                     help='run name; the log directory is logs/<name>-<jobid or time>')
    run.add_argument('--log-dir', type=Path, default=REPO / 'logs')
    run.add_argument('--log-every', type=int, default=100)
    run.add_argument('--val-every', type=int, default=2000)
    run.add_argument('--val-batches', type=int, default=64)
    run.add_argument('--checkpoint-minutes', type=float, default=20.0)
    run.add_argument('--resume', default='auto',
                     help="'auto' to continue logs/<name>-*/checkpoints/last.pt, "
                          "'none', or a path to a checkpoint")
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--no-compile', action='store_true')
    run.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    run.add_argument('--verify-batches', type=int, default=2,
                     help='batches to check the rebuilt visibility mask on at startup')
    run.add_argument('--smoke', action='store_true',
                     help='tiny everything - a few shards, a few steps, no RAM cache. '
                          'Runs on a login node and proves the pipeline end to end.')

    args = parser.parse_args(argv)

    if args.smoke:
        args.max_shards = args.max_shards or 8
        args.val_shards = min(args.val_shards, 2)
        args.workers = min(args.workers, 2)
        args.batch_size = min(args.batch_size, 1024)
        args.max_steps = args.max_steps or 30
        args.log_every = 10
        args.val_every = 20
        args.val_batches = 4
        args.warmup_steps = 5
        args.no_cache = True
        args.no_compile = True
        args.hours = min(args.hours, 0.25)
        # Somewhere a real run's --resume auto cannot glob into: its pattern is
        # logs/<name>-*, and 'fow-smoke-<stamp>' matches 'fow-*'.
        args.log_dir = Path(args.log_dir) / 'smoke'

    return args


# --------------------------------------------------------------------------
# Where a run writes
# --------------------------------------------------------------------------

class RunLog:
    '''The ``logs/<run>/`` directory and everything written into it.'''

    def __init__(self, args):
        stamp = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d-%H%M%S')
        self.directory = Path(args.log_dir) / f'{args.name}-{stamp}'
        self.checkpoints = self.directory / 'checkpoints'
        self.checkpoints.mkdir(parents=True, exist_ok=True)

        self.log = logging.getLogger('fow')
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()

        form = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s', '%H:%M:%S')
        for handler in (
            logging.FileHandler(self.directory / 'train.log'),
            logging.StreamHandler(sys.stdout),
        ):
            handler.setFormatter(form)
            self.log.addHandler(handler)

        self._writers = {}
        self.write_json('config.json', vars(args))

    def write_json(self, name, payload):
        (self.directory / name).write_text(json.dumps(payload, indent=2, default=str))

    def row(self, name, record):
        '''Append one record to ``<name>.csv``, writing the header the first time.'''
        if name not in self._writers:
            handle = open(self.directory / f'{name}.csv', 'a', newline='')
            writer = csv.DictWriter(handle, fieldnames=list(record))

            if handle.tell() == 0:
                writer.writeheader()

            self._writers[name] = (handle, writer)

        handle, writer = self._writers[name]
        writer.writerow(record)
        handle.flush()

    def close(self):
        for handle, _ in self._writers.values():
            handle.close()


def environment():
    '''What this run is running on, for when two runs disagree.'''
    def git(*command):
        try:
            return subprocess.run(
                ['git', *command], cwd=REPO, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return 'unknown'

    facts = {
        'host': socket.gethostname(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'commit': git('rev-parse', 'HEAD'),
        'dirty': bool(git('status', '--porcelain')),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'cpus': os.cpu_count(),
    }

    if torch.cuda.is_available():
        facts['gpu'] = torch.cuda.get_device_name(0)
        facts['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        )
        facts['cuda'] = torch.version.cuda

    return facts


# --------------------------------------------------------------------------
# Stopping politely
# --------------------------------------------------------------------------

class Interrupted:
    '''True once SLURM (or a person) has asked the job to wind up.

    SIGUSR1 is what ``--signal=B:USR1@300`` sends five minutes before the
    walltime runs out; SIGTERM is what arrives when a job is cancelled or
    requeued.  Either way there is time to finish the step and write a
    checkpoint, which is all this needs to survive.
    '''

    def __init__(self, log):
        self.raised = False
        self.log = log

        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(number, self._catch)

    def _catch(self, number, _frame):
        self.log.warning(f'caught {signal.Signals(number).name} - '
                         f'finishing the step and checkpointing')
        self.raised = True

    def __bool__(self):
        return self.raised


# --------------------------------------------------------------------------
# Learning rate
# --------------------------------------------------------------------------

def learning_rate(args, step, progress):
    '''Linear warmup, then cosine down to ``--min-lr-frac`` of ``--lr``.

    ``progress`` is how far through the budget the run is - elapsed over
    ``--hours`` when ``--schedule time``, which is the honest measure for a job
    that is bounded by a clock rather than by a number of steps.
    '''
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps

    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))

    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


def parameter_groups(model, weight_decay):
    '''Decay the matrices, leave the biases and norms alone.'''
    decayed = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    plain = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    return [
        {'params': decayed, 'weight_decay': weight_decay},
        {'params': plain, 'weight_decay': 0.0},
    ]


# --------------------------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------------------------

@dataclass
class Totals:
    '''Running sums, kept on the GPU so that a step never has to synchronise.

    Every metric is a sum and a count; the ratio is only taken when something is
    about to be written down, which is once every ``--log-every`` steps rather
    than once a step.
    '''

    device: str
    n: int = 12

    def __post_init__(self):
        self.sums = torch.zeros(self.n, dtype=torch.float64, device=self.device)

    def reset(self):
        self.sums.zero_()

    def add(self, values):
        # detach() so that a caller handing over a sum that is still attached to
        # the graph cannot make these running totals keep that graph alive.
        self.sums += torch.stack(values).detach().to(torch.float64)

    def take(self):
        got = self.sums.tolist()
        self.reset()
        return got


def step_metrics(logits, target, hidden):
    '''Loss to backpropagate, plus the sums :class:`Totals` accumulates.

    Loss is over every square, hidden or not.  The visible ones are free - the
    answer is in the input - but they are what anchors ``e_piece`` and
    ``e_side`` on a meaning, and leaving them out would make the model's
    prediction for a square it can see untrained and therefore useless to
    anything that wants ``P(piece on square)`` everywhere.  ``--hidden-weight``
    is the dial if the free half starts crowding the hard half out.
    '''
    flat_logits = logits.reshape(-1, N_CLASSES).float()
    flat_target = target.reshape(-1)
    flat_hidden = hidden.reshape(-1)

    losses = F.cross_entropy(flat_logits, flat_target, reduction='none')

    # The sums are read, never backpropagated, and they are built under
    # no_grad because otherwise they are built under the autograd graph: a sum
    # like ``(losses * flat_hidden).sum()`` is a graph node of its own, off the
    # path ``loss.backward()`` walks, so nothing ever frees the activations it
    # saved.  Totals then holds one such node per step and a 49k-parameter
    # model runs an 80 GB card out of memory in a few thousand steps.
    with torch.no_grad():
        predicted = flat_logits.argmax(-1)
        right = predicted == flat_target
        occupied = flat_target != CLASS_EMPTY

        squares = flat_target.numel()
        n_hidden = flat_hidden.sum()
        n_hidden_piece = (flat_hidden & occupied).sum()

        # Cross entropy is -log P(true class), so the probability the model put
        # on the right answer is already computed - no second softmax over
        # (batch * 64, 13) just to report it.
        probability = torch.exp(-losses)

        sums = [
            losses.sum(), torch.tensor(float(squares), device=losses.device),
            (losses * flat_hidden).sum(), n_hidden.float(),
            right.sum().float(),
            (right & flat_hidden).sum().float(),
            (right & flat_hidden & occupied).sum().float(), n_hidden_piece.float(),
            (probability * flat_hidden).sum(),
            (flat_hidden & ~occupied).sum().float(),
            torch.tensor(0.0, device=losses.device),
            torch.tensor(0.0, device=losses.device),
        ]

    return losses, sums


def weighted_loss(losses, hidden, hidden_weight):
    if hidden_weight == 1.0:
        return losses.mean()

    weights = torch.where(hidden.reshape(-1), hidden_weight, 1.0)
    return (losses * weights).sum() / weights.sum()


def summarise(sums):
    '''The twelve running sums as the numbers a person reads.'''
    (loss, squares, hidden_loss, hidden, right, hidden_right,
     hidden_piece_right, hidden_piece, probability, hidden_empty, _, _) = sums

    safe = lambda a, b: a / b if b else float('nan')

    return {
        'loss': safe(loss, squares),
        'hidden_loss': safe(hidden_loss, hidden),
        'acc': safe(right, squares),
        'hidden_acc': safe(hidden_right, hidden),
        'hidden_piece_acc': safe(hidden_piece_right, hidden_piece),
        'hidden_p_true': safe(probability, hidden),
        'hidden_frac': safe(hidden, squares),
        'baseline': safe(hidden_empty, hidden),
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, args, device, encode):
    '''One pass over held-out shards, with a per-piece breakdown.

    The breakdown is recall on hidden squares - of the black knights that were
    hidden, how many did the model put in the right place - because that is what
    the model is for, and a single accuracy hides it completely: knights and
    rooks that have not moved are easy, and a queen halfway up the board is not.
    '''
    model.eval()
    totals = Totals(device)
    found = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
    present = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)

    seen = 0
    for batch in loader:
        view, truth, en_passant = (t.to(device, non_blocking=True) for t in batch)
        fields = encode(view, truth, en_passant)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(fields['piece'], fields['side'], fields['visibility'])

        _, sums = step_metrics(logits, fields['target'], fields['hidden'])
        totals.add(sums)

        target = fields['target'].reshape(-1)
        hidden = fields['hidden'].reshape(-1)
        right = logits.reshape(-1, N_CLASSES).float().argmax(-1) == target

        present.scatter_add_(0, target, hidden.to(present.dtype))
        found.scatter_add_(0, target, (hidden & right).to(found.dtype))

        seen += 1
        if seen >= args.val_batches:
            break

    model.train()

    recall = (found / present.clamp(min=1)).tolist()
    counts = present.tolist()

    report = summarise(totals.take())
    report['val_batches'] = seen

    for index, name in enumerate(CLASS_NAMES):
        if index == CLASS_EMPTY:
            continue
        report[f'recall_{name}'] = recall[index]
        report[f'hidden_{name}'] = int(counts[index])

    return report


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def save_checkpoint(path, model, optimiser, step, elapsed, positions, best, args):
    core = getattr(model, '_orig_mod', model)

    tmp = path.with_suffix('.tmp')
    torch.save(
        {
            'model': core.state_dict(),
            'config': core.config.to_dict(),
            'optimiser': optimiser.state_dict(),
            'step': step,
            'elapsed': elapsed,
            'positions': positions,
            'best': best,
            'args': vars(args),
            'torch_rng': torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def find_resume(args):
    '''The checkpoint ``--resume`` points at, or ``None``.'''
    if args.resume in ('none', '', None):
        return None

    if args.resume != 'auto':
        return Path(args.resume)

    candidates = sorted(
        Path(args.log_dir).glob(f'{args.name}-*/checkpoints/last.pt'),
        key=lambda path: path.stat().st_mtime,
    )

    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    run = RunLog(args)
    log = run.log
    stop = Interrupted(log)

    torch.manual_seed(args.seed)
    device = args.device

    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    run.write_json('env.json', environment())
    log.info(f'run directory: {run.directory}')
    log.info(f'environment: {json.dumps(environment())}')

    # ---- data ----------------------------------------------------------
    shards = pp.list_shards(args.shards)
    if args.max_shards:
        shards = shards[:args.max_shards]

    train_paths, val_paths = pp.split_shards(shards, args.val_shards, args.split_seed)
    log.info(f'{len(shards):,} shards: {len(train_paths):,} train, {len(val_paths):,} validation')

    cache = pp.ShardBytes(
        shards, cache=not args.no_cache, threads=args.cache_threads, log=log.info
    )
    index = {path: i for i, path in enumerate(shards)}
    subset = lambda paths: pp.ShardSubset(cache, [index[path] for path in paths])

    train_loader = pp.positions_loader(
        subset(train_paths), args.batch_size, args.workers,
        pool_shards=args.pool_shards, seed=args.seed, passes=None,
        pin_memory=device == 'cuda',
    )
    val_shards = subset(val_paths)

    def val_loader():
        return pp.positions_loader(
            val_shards, args.batch_size, min(args.workers, 4),
            pool_shards=args.pool_shards, seed=12345, passes=1,
            pin_memory=device == 'cuda',
        )

    # ---- model ---------------------------------------------------------
    config = ModelConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout,
    )
    model = FogOfWarNet(config).to(device)

    log.info(f'model: {config}')
    log.info(f'parameters: {parameter_count(model):,}  {parameter_breakdown(model)}')
    run.write_json('model.json', {
        'config': config.to_dict(),
        'parameters': parameter_count(model),
        'breakdown': parameter_breakdown(model),
    })

    optimiser = torch.optim.AdamW(
        parameter_groups(model, args.weight_decay),
        lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
        fused=device == 'cuda',
    )

    step, elapsed, positions, best = 0, 0.0, 0, float('inf')
    checkpoint = find_resume(args)

    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        step, elapsed = state['step'], state['elapsed']
        positions, best = state['positions'], state['best']
        torch.set_rng_state(state['torch_rng'].cpu())
        log.info(f'resumed {checkpoint} at step {step:,}, {elapsed / 3600:.2f} h elapsed')
    else:
        log.info('starting from scratch')

    encode = pp.encode_batch
    if not args.no_compile:
        try:
            model = torch.compile(model)
            encode = torch.compile(pp.encode_batch, dynamic=False)
            log.info('compiled model and encoder')
        except Exception as error:                        # pragma: no cover
            log.warning(f'torch.compile unavailable ({error}); running eager')

    # ---- a last check that the mask matches the data it is paired with --
    if args.verify_batches:
        checked = 0
        for batch in val_loader():
            view, truth, en_passant = (t.to(device) for t in batch)
            wrong = pp.check_consistency(view, truth, en_passant)
            if wrong:
                raise RuntimeError(
                    f'rebuilt visibility mask disagrees with the recorded view on '
                    f'{wrong} squares - see src/preprocess.visibility_mask'
                )
            checked += 1
            if checked >= args.verify_batches:
                break
        log.info(f'visibility mask verified on {checked * args.batch_size:,} positions')

    # ---- train ---------------------------------------------------------
    budget = args.hours * 3600
    totals = Totals(device)
    started = time.monotonic()
    last_checkpoint = time.monotonic()
    window_started = time.monotonic()
    window_wait = 0.0
    window_steps = 0

    log.info(f'training for {args.hours:.2f} h '
             f'({args.batch_size:,} positions/step, {args.schedule} schedule)')

    batches = iter(train_loader)
    model.train()

    while True:
        now = time.monotonic()
        total_elapsed = elapsed + (now - started)

        if total_elapsed >= budget:
            log.info(f'time budget reached at step {step:,}')
            break
        if args.max_steps and step >= args.max_steps:
            log.info(f'step budget reached at step {step:,}')
            break
        if stop:
            break

        waiting = time.monotonic()
        view, truth, en_passant = next(batches)
        window_wait += time.monotonic() - waiting

        view = view.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        en_passant = en_passant.to(device, non_blocking=True)
        fields = encode(view, truth, en_passant)

        progress = (
            total_elapsed / budget if args.schedule == 'time'
            else (step / args.max_steps if args.max_steps else 0.0)
        )
        rate = learning_rate(args, step, progress)
        for group in optimiser.param_groups:
            group['lr'] = rate

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(fields['piece'], fields['side'], fields['visibility'])

        losses, sums = step_metrics(logits, fields['target'], fields['hidden'])
        loss = weighted_loss(losses, fields['hidden'], args.hidden_weight)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()

        sums[-2] = norm.detach()
        sums[-1] = torch.ones((), device=device)
        totals.add(sums)

        step += 1
        window_steps += 1
        positions += args.batch_size

        # ---- periodic logging ----
        if step % args.log_every == 0:
            got = totals.take()
            record = summarise(got)
            window = time.monotonic() - window_started

            record.update({
                'step': step,
                'hours': round(total_elapsed / 3600, 4),
                'positions': positions,
                'lr': rate,
                'grad_norm': got[-2] / max(got[-1], 1),
                'pos_per_s': window_steps * args.batch_size / max(window, 1e-9),
                'data_wait_frac': window_wait / max(window, 1e-9),
            })
            if device == 'cuda':
                record['gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30

            record = {'step': record.pop('step'), **record}
            run.row('metrics', record)
            run.write_json('progress.json', record)

            log.info(
                f"step {step:>8,} | {total_elapsed / 3600:5.2f} h | "
                f"loss {record['loss']:.4f} | hidden {record['hidden_loss']:.4f} | "
                f"hid acc {record['hidden_acc']:.4f} (base {record['baseline']:.4f}) | "
                f"piece acc {record['hidden_piece_acc']:.4f} | "
                f"lr {rate:.2e} | {record['pos_per_s'] / 1e3:,.0f}k pos/s | "
                f"wait {record['data_wait_frac']:.1%}"
            )

            window_started = time.monotonic()
            window_wait = 0.0
            window_steps = 0

        # ---- validation ----
        if step % args.val_every == 0:
            report = validate(model, val_loader(), args, device, encode)
            report = {'step': step, 'hours': round(total_elapsed / 3600, 4), **report}
            run.row('val', report)

            log.info(
                f"  val {step:>8,} | loss {report['loss']:.4f} | "
                f"hidden {report['hidden_loss']:.4f} | "
                f"hid acc {report['hidden_acc']:.4f} (base {report['baseline']:.4f}) | "
                f"piece acc {report['hidden_piece_acc']:.4f} | "
                f"P(true) {report['hidden_p_true']:.4f}"
            )
            log.info(
                '  recall  ' + '  '.join(
                    f'{name}:{report[f"recall_{name}"]:.3f}'
                    for name in CLASS_NAMES if name != '.'
                )
            )

            if report['hidden_loss'] < best:
                best = report['hidden_loss']
                save_checkpoint(run.checkpoints / 'best.pt', model, optimiser,
                                step, total_elapsed, positions, best, args)
                log.info(f'  new best hidden loss {best:.4f} -> best.pt')

            window_started = time.monotonic()
            window_wait = 0.0
            window_steps = 0

        # ---- checkpoint on a timer ----
        if (time.monotonic() - last_checkpoint) / 60 >= args.checkpoint_minutes:
            save_checkpoint(run.checkpoints / 'last.pt', model, optimiser,
                            step, total_elapsed, positions, best, args)
            last_checkpoint = time.monotonic()
            log.info(f'  checkpointed at step {step:,}')

    # ---- wind up -------------------------------------------------------
    total_elapsed = elapsed + (time.monotonic() - started)
    save_checkpoint(run.checkpoints / 'last.pt', model, optimiser,
                    step, total_elapsed, positions, best, args)

    report = validate(model, val_loader(), args, device, encode)
    report = {'step': step, 'hours': round(total_elapsed / 3600, 4), **report}
    run.row('val', report)
    run.write_json('final.json', report)

    log.info(f'finished: {step:,} steps, {positions:,} positions, '
             f'{total_elapsed / 3600:.2f} h')
    log.info(f"final hidden loss {report['hidden_loss']:.4f}, "
             f"hidden accuracy {report['hidden_acc']:.4f} "
             f"against a {report['baseline']:.4f} baseline")
    log.info(f'checkpoints in {run.checkpoints}')

    run.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
