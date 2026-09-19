'''Train :class:`~src.regressive.RegressiveFogOfWar` on windows of plies.

A sibling of ``src/train.py`` rather than a flag inside it.  The run shape, the
logging, the checkpoint-and-requeue dance and every metric are imported from
there unchanged - what differs is only the three things that have to: the loader
yields windows, the model takes a time axis, and the encoder folds that axis
into the batch before handing the boards to the tested visibility code.

Keeping it separate is also what lets this be written while the 10M and 50M
single-position runs are still going: they re-import ``src/train.py`` if SLURM
requeues them, and a run fifteen hours deep is not the place to discover an
edit.

The budget
----------

``ou_bcs_normal`` caps at 24 hours of wall clock, so ``--hours 24`` cannot fit
in one allocation: the RAM cache and ``torch.compile`` have to come out of it
first.  That is what ``--requeue`` is for.  The job takes SIGUSR1 five minutes
before the walltime, writes ``last.pt`` and exits; SLURM requeues it under the
same job id, ``--resume auto`` restores the elapsed clock along with the
weights, and the last ~40 minutes of the budget run in the second allocation.
The cosine schedule is annealed against the restored clock, so it spans the
true 24 hours rather than restarting.

What this does and does not fix
-------------------------------

It attacks the *information* limit - the single-position model cannot know what
it saw ten plies ago - and the piece it should help most is the rook, worst
located of the six at 0.32 top-1.

It does not attack the *coherence* limit.  The head is still 64 independent
softmaxes, so the marginals will stay well calibrated and the joint will stay
unrepresentable: sample every hidden square independently and you still get
exactly one black king about 60% of the time.  That needs a head that couples
the squares, which is a different model, not a different input.
'''

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from src import preprocess as pp
from src.history import encode_histories, histories_loader
from src.model import CLASS_EMPTY, CLASS_NAMES, N_CLASSES
from src.regressive import (
    DEFAULT_HISTORY,
    RegressiveConfig,
    RegressiveFogOfWar,
    parameter_breakdown,
    parameter_count,
)
from src.train import (
    Interrupted,
    RunLog,
    Totals,
    environment,
    find_resume,
    learning_rate,
    parameter_groups,
    save_checkpoint,
    step_metrics,
    summarise,
    weighted_loss,
)

REPO = Path(__file__).resolve().parent.parent

#: Exit code for "the walltime ran out before --hours did".  Anything the shell
#: would not otherwise produce; scripts/train_regressive.sbatch requeues on it,
#: because SLURM's own --requeue does not cover a process that exited 0.
UNFINISHED = 42


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train the history model on windows of consecutive plies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group('data')
    data.add_argument('--shards', default=os.environ.get('FOW_PROCESSED', str(pp.PROCESSED_DIR)))
    data.add_argument('--max-shards', type=int, default=0)
    data.add_argument('--val-shards', type=int, default=32)
    data.add_argument('--split-seed', type=int, default=0)
    data.add_argument('--workers', type=int, default=24)
    data.add_argument('--pool-shards', type=int, default=4)
    data.add_argument('--no-cache', action='store_true')
    data.add_argument('--cache-threads', type=int, default=16)

    shape = parser.add_argument_group('model')
    shape.add_argument('--history', type=int, default=DEFAULT_HISTORY,
                       help='plies fed in, counting the one being predicted')
    shape.add_argument('--d-model', type=int, default=320)
    shape.add_argument('--n-heads', type=int, default=4)
    shape.add_argument('--n-layers', type=int, default=6)
    shape.add_argument('--d-ff', type=int, default=1280)
    shape.add_argument('--dropout', type=float, default=0.0)

    # The batch is in *positions*; each one drags --history plies of context in
    # with it, so a step is batch * history * 64 tokens.  1024 x 8 matches the
    # 10M single-position run's 8192 x 1 token for token - and, because the
    # time axis is causal and every ply is a target, square for supervised
    # square as well.  That is what lets the learning rate be the 10M run's
    # rather than something derated for a smaller effective batch.
    fit = parser.add_argument_group('optimiser')
    fit.add_argument('--batch-size', type=int, default=1024)
    fit.add_argument('--lr', type=float, default=7e-4)
    fit.add_argument('--min-lr-frac', type=float, default=0.05)
    fit.add_argument('--warmup-steps', type=int, default=2000)
    fit.add_argument('--weight-decay', type=float, default=0.01)
    fit.add_argument('--beta1', type=float, default=0.9)
    fit.add_argument('--beta2', type=float, default=0.95)
    fit.add_argument('--grad-clip', type=float, default=1.0)
    fit.add_argument('--hidden-weight', type=float, default=1.0)

    run = parser.add_argument_group('run')
    run.add_argument('--schedule', choices=('time', 'steps'), default='time')
    run.add_argument('--hours', type=float, default=24.0)
    run.add_argument('--max-steps', type=int, default=0)
    run.add_argument('--name', default='fowhist')
    run.add_argument('--log-dir', default=str(REPO / 'logs'))
    run.add_argument('--log-every', type=int, default=100)
    run.add_argument('--val-every', type=int, default=2000)
    run.add_argument('--val-batches', type=int, default=64)
    run.add_argument('--checkpoint-minutes', type=float, default=20.0)
    run.add_argument('--resume', default='auto')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--no-compile', action='store_true')
    run.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    run.add_argument('--verify-batches', type=int, default=2)
    run.add_argument('--smoke', action='store_true')

    args = parser.parse_args(argv)

    if args.smoke:
        args.max_shards = args.max_shards or 8
        args.val_shards = min(args.val_shards, 2)
        args.workers = min(args.workers, 2)
        args.batch_size = min(args.batch_size, 128)
        args.max_steps = args.max_steps or 30
        args.log_every = 10
        args.val_every = 20
        args.val_batches = 4
        args.warmup_steps = 5
        args.no_cache = True
        args.no_compile = True
        args.hours = min(args.hours, 0.25)
        args.log_dir = Path(args.log_dir) / 'smoke'

    return args


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, args, device):
    '''``src.train.validate``, but the batches carry a time axis.'''
    model.eval()
    totals = Totals(device)
    found = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
    present_count = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)

    seen = 0
    for batch in loader:
        view, truth, en_passant = (t.to(device, non_blocking=True) for t in batch)
        fields = encode_histories(view, truth, en_passant)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(fields['piece'], fields['side'], fields['visibility'])

        # The newest ply alone.  Training uses all T of them, but the number
        # worth writing down is the one the single-position runs report: the
        # belief about the position white is actually standing in, with a full
        # window behind it.  Averaging in the earlier plies would flatter the
        # model on the ones that have barely any history and make val.csv
        # incomparable with logs/fow*-*/val.csv.
        logits = logits[:, -1]
        fields = {k: v[:, -1] for k, v in fields.items()}

        _, sums = step_metrics(logits, fields['target'], fields['hidden'])
        totals.add(sums)

        target = fields['target'].reshape(-1)
        hidden = fields['hidden'].reshape(-1)
        right = logits.reshape(-1, N_CLASSES).float().argmax(-1) == target

        present_count.scatter_add_(0, target, hidden.to(present_count.dtype))
        found.scatter_add_(0, target, (hidden & right).to(found.dtype))

        seen += 1
        if seen >= args.val_batches:
            break

    model.train()

    recall = (found / present_count.clamp(min=1)).tolist()
    counts = present_count.tolist()

    report = summarise(totals.take())
    report['val_batches'] = seen

    for index, name in enumerate(CLASS_NAMES):
        if index == CLASS_EMPTY:
            continue
        report[f'recall_{name}'] = recall[index]
        report[f'hidden_{name}'] = int(counts[index])

    return report


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

    # --max-shards at or below --val-shards hands every shard to validation and
    # leaves the training loader with nothing to yield.  That is not an error
    # anywhere downstream: the job simply blocks on next(batches) for its whole
    # allocation, and a hang looks exactly like a slow filesystem.
    if not train_paths or not val_paths:
        raise SystemExit(
            f'need shards on both sides of the split, got {len(train_paths):,} '
            f'train and {len(val_paths):,} validation from {len(shards):,} '
            f'shards - lower --val-shards or raise --max-shards'
        )

    cache = pp.ShardBytes(
        shards, cache=not args.no_cache, threads=args.cache_threads, log=log.info
    )
    index = {path: i for i, path in enumerate(shards)}
    subset = lambda paths: pp.ShardSubset(cache, [index[path] for path in paths])

    train_loader = histories_loader(
        subset(train_paths), args.batch_size, args.history, args.workers,
        pool_shards=args.pool_shards, seed=args.seed, passes=None,
        pin_memory=device == 'cuda',
    )
    val_subset = subset(val_paths)

    def val_loader():
        return histories_loader(
            val_subset, args.batch_size, args.history, min(args.workers, 4),
            pool_shards=args.pool_shards, seed=12345, passes=1,
            pin_memory=device == 'cuda',
        )

    # ---- model ---------------------------------------------------------
    config = RegressiveConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, history=args.history,
    )
    model = RegressiveFogOfWar(config).to(device)

    log.info(f'model: {config}')
    log.info(f'parameters: {parameter_count(model):,}  {parameter_breakdown(model)}')
    log.info(f'tokens per step: {args.batch_size * args.history * 64:,}, '
             f'all of them supervised')
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

    if not args.no_compile:
        try:
            model = torch.compile(model)
            log.info('compiled model')
        except Exception as error:                        # pragma: no cover
            log.warning(f'torch.compile unavailable ({error}); running eager')

    # ---- the mask still has to match the view it is paired with --------
    # Stricter than the single-position check: every ply of the window is
    # verified, not just the one being predicted.
    if args.verify_batches:
        checked = 0
        for batch in val_loader():
            view, truth, en_passant = (t.to(device) for t in batch)
            plies = view.shape[1]
            wrong = pp.check_consistency(
                view.reshape(-1, 8, 8),
                truth.reshape(-1, 8, 8),
                en_passant.reshape(-1),
            )
            if wrong:
                raise RuntimeError(
                    f'rebuilt visibility mask disagrees with the recorded view on '
                    f'{wrong} squares - see src/preprocess.visibility_mask'
                )
            checked += 1
            if checked >= args.verify_batches:
                break
        log.info(f'visibility mask verified on '
                 f'{checked * args.batch_size * plies:,} boards '
                 f'({checked * args.batch_size:,} windows)')

    # ---- train ---------------------------------------------------------
    budget = args.hours * 3600
    totals = Totals(device)
    started = time.monotonic()
    last_checkpoint = time.monotonic()
    window_started = time.monotonic()
    window_wait = 0.0
    window_steps = 0

    log.info(f'training for {args.hours:.2f} h '
             f'({args.batch_size:,} positions/step, {args.history} plies each, '
             f'{args.schedule} schedule)')

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
        fields = encode_histories(view, truth, en_passant)

        progress = (
            total_elapsed / budget if args.schedule == 'time'
            else (step / args.max_steps if args.max_steps else 0.0)
        )
        rate = learning_rate(args, step, progress)
        for group in optimiser.param_groups:
            group['lr'] = rate

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(fields['piece'], fields['side'], fields['visibility'])

        # Every ply in the window is a target, so the flat (batch * T * 64)
        # view of these is what the metrics run over - step_metrics reshapes to
        # (-1, N_CLASSES) and does not care how many axes came before.
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
                f"lr {rate:.2e} | {record['pos_per_s'] / 1e3:,.1f}k pos/s | "
                f"wait {record['data_wait_frac']:.1%}"
            )

            window_started = time.monotonic()
            window_wait = 0.0
            window_steps = 0

        # ---- validation ----
        if step % args.val_every == 0:
            report = validate(model, val_loader(), args, device)
            report = {'step': step, 'hours': round(total_elapsed / 3600, 4), **report}
            run.row('val', report)

            log.info(
                f"  validation | hidden {report['hidden_loss']:.4f} | "
                f"hid acc {report['hidden_acc']:.4f} | "
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
    # Whether the budget was actually spent, or the job was cut short.  The
    # difference matters twice: `final.json` should not appear on a run that
    # has not finished - it is what every comparison reads - and the walltime
    # here is 24h against a 24h budget, so the last few minutes only happen if
    # scripts/train_regressive.sbatch knows to requeue.  SLURM will not do it
    # on its own: --requeue covers preemption and node failure, not a process
    # that caught SIGUSR1 and exited 0.
    total_elapsed = elapsed + (time.monotonic() - started)
    spent = total_elapsed >= budget or (args.max_steps and step >= args.max_steps)

    save_checkpoint(run.checkpoints / 'last.pt', model, optimiser,
                    step, total_elapsed, positions, best, args)

    report = validate(model, val_loader(), args, device)
    report = {'step': step, 'hours': round(total_elapsed / 3600, 4), **report}
    run.row('val', report)
    run.write_json('final.json' if spent else 'interrupted.json', report)

    log.info(f'{"finished" if spent else "interrupted"} at step {step:,} '
             f'after {total_elapsed / 3600:.2f} h of {args.hours:.2f} h')
    log.info(f'hidden loss {report["hidden_loss"]:.4f}, '
             f'hidden acc {report["hidden_acc"]:.4f}, '
             f'piece acc {report["hidden_piece_acc"]:.4f}')

    if not spent:
        log.info(f'{(budget - total_elapsed) / 60:.0f} min of budget left - '
                 f'exiting {UNFINISHED} so the batch script can requeue')

    run.close()

    return report if spent else None


if __name__ == '__main__':
    raise SystemExit(0 if main() is not None else UNFINISHED)
