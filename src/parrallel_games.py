'''Run the fog-of-war pipeline across cores - one parquet file per job.

The job is embarrassingly parallel and the profile says why: playing the moves
is nearly free, and ~94% of the time goes into ``ChessState.gen_state`` working
out what each side can see.  That is pure CPU on one game at a time, with no
state shared between games, so the only thing standing between it and a linear
speedup is how the work is handed out.

One parquet file per worker
    A file is ~1,000,000 games and several hours of work, so the unit is large
    enough that start-up costs vanish, and self-contained enough that workers
    never talk to each other.  Handing out single *games* instead would mean
    shipping ~45 KB of arrays back to the parent per game, which costs more than
    the game took to generate.

Workers write their own shards
    Nothing is sent back to the parent except a line of statistics.
    ``ChessGame.save_data`` names shards ``<file stem>-<shard>.pt``, and every
    worker owns a different file stem, so two workers cannot collide on a
    filename no matter how the shards line up.  That is the property that lets
    this run without a lock.

One core each
    Each worker is pinned to a single core where the OS supports it, and the
    BLAS/OpenMP thread pools are held to one thread apiece.  Without that, eight
    workers each spin up their own thread pool, oversubscribe the machine and
    run slower than four would.

Resumable
    A run over the whole corpus is long enough that it will be interrupted.
    Files that already have shards on disk are skipped unless ``--overwrite``
    says otherwise, so restarting picks up roughly where it stopped.

Usage::

    python -m src.parrallel_games --workers 8
    python -m src.parrallel_games --workers 16 --output /scratch/fog --shard-size 2000
    python -m src.parrallel_games --limit 200 --dry-run
'''

# These have to be set before numpy, torch or pandas are imported anywhere, and
# 'spawn' re-imports this module in each child, so the children inherit them
# from this block rather than from the parent's environment.
import os

for _var in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_var, '1')

import argparse
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

#: Set in each worker by ``init_worker``; only used to pick a core to pin to.
WORKER_ID = 0


# --------------------------------------------------------------------------
# Worker setup
# --------------------------------------------------------------------------

def pin_to_core(index):
    '''Pin this process to one core, on platforms that can express it.

    Linux only in practice - macOS has no ``sched_setaffinity`` - so this is a
    no-op on a laptop and does the real thing on the server.  Returns the core
    it pinned to, or ``None`` if it could not.
    '''
    if not hasattr(os, 'sched_setaffinity'):
        return None

    available = sorted(os.sched_getaffinity(0))
    core = available[index % len(available)]
    os.sched_setaffinity(0, {core})

    return core


def worker_index():
    '''A small distinct number for this worker process.

    ``ProcessPoolExecutor`` does not number its workers, and a shared counter
    would mean handing a synchronized object to every child - which is exactly
    the kind of thing that behaves differently under 'spawn'.  multiprocessing
    already names workers ``SpawnProcess-1``, ``SpawnProcess-2``, ... as it
    starts them, and that trailing number is distinct among the live workers,
    which is all a core assignment needs.
    '''
    tail = multiprocessing.current_process().name.rsplit('-', 1)[-1]
    return int(tail) - 1 if tail.isdigit() else 0


def init_worker(pin):
    '''Give this worker a number, and hold it to one core and one thread.'''
    global WORKER_ID

    WORKER_ID = worker_index()

    if pin:
        pin_to_core(WORKER_ID)

    # numpy and pandas respect the environment set at the top of this module;
    # torch has to be told separately, and only once it has been imported.
    import torch
    torch.set_num_threads(1)


# --------------------------------------------------------------------------
# The job
# --------------------------------------------------------------------------

def shards_of(source, out_dir):
    '''The shards already on disk for one parquet file.'''
    return sorted(out_dir.glob(f'{source.stem}-*.pt'))


def process_file(source, out_dir, shard_size, limit):
    '''Play every game in one parquet file and write its shards.

    Runs in a worker process, so it imports the pipeline itself rather than
    inheriting it: under 'spawn' nothing is carried over from the parent.

    A game that will not replay is skipped rather than allowed to kill a job
    several hours in.  That is safe to do here because ``run_game`` builds its
    arrays locally and only appends them to the buffer once the whole game has
    played, so a game that raises part way through leaves nothing behind.
    '''
    from src import data
    from src import game

    source = Path(source)
    out_dir = Path(out_dir)

    data.output_dir = out_dir
    game.TENSOR_SIZE = shard_size

    pipeline = data.ChessGame([source])

    played = skipped = plies = 0
    first_error = None
    started = time.perf_counter()

    for i, moves in enumerate(pipeline.games):
        if limit is not None and i >= limit:
            break

        try:
            pipeline.state = data.ChessState()
            pipeline.run_game(moves)
        except Exception as exc:
            skipped += 1
            if first_error is None:
                first_error = f'game {i}: {type(exc).__name__}: {exc}'
            continue

        played += 1
        plies += len(moves)

        if len(pipeline.input_data) >= game.TENSOR_SIZE:
            pipeline.save_data()

    # The tail of the file is a short final shard, not something to drop
    if pipeline.input_data:
        pipeline.save_data()

    return {
        'file': source.name,
        'worker': WORKER_ID,
        'core': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'played': played,
        'skipped': skipped,
        'plies': plies,
        'shards': len(shards_of(source, out_dir)),
        'seconds': time.perf_counter() - started,
        'first_error': first_error,
    }


# --------------------------------------------------------------------------
# Driving it
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the fog-of-war pipeline across cores, one parquet file per job.'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='directory of parquet files (default: the one main.DATA_DIR names)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='where shards are written (default: <data-dir>/processed)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='worker processes, one core each (default: every core on the machine)',
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=None,
        help='samples per shard, two per game (default: game.TENSOR_SIZE)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='stop after this many games per file, for a smoke test',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='reprocess files that already have shards instead of skipping them',
    )
    parser.add_argument(
        '--no-affinity',
        action='store_true',
        help='do not pin workers to cores (leave placement to the OS)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print the plan and exit without playing anything',
    )

    return parser.parse_args(argv)


def plan(args):
    '''``(files to do, files skipped, output directory)`` for this run.'''
    from main import DATA_DIR
    from src import game

    data_dir = args.data_dir or DATA_DIR
    out_dir = args.output or (data_dir / 'processed')

    sources = sorted(Path(data_dir).glob('*.parquet'))
    if not sources:
        raise SystemExit(f'no parquet files in {data_dir}')

    if args.overwrite:
        return sources, [], out_dir, args.shard_size or game.TENSOR_SIZE

    todo, done = [], []
    for source in sources:
        (done if shards_of(source, Path(out_dir)) else todo).append(source)

    return todo, done, out_dir, args.shard_size or game.TENSOR_SIZE


def main(argv=None):
    args = parse_args(argv)
    todo, done, out_dir, shard_size = plan(args)

    workers = args.workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(todo))) if todo else 1

    print(f'files to process : {len(todo)}')
    print(f'already done     : {len(done)} (use --overwrite to redo)')
    print(f'workers          : {workers} ({"pinned" if not args.no_affinity else "unpinned"})')
    print(f'output           : {out_dir}')
    print(f'shard size       : {shard_size} samples ({shard_size // 2} games)')
    if args.limit:
        print(f'limit            : {args.limit} games per file')

    if args.dry_run:
        for source in todo:
            print(f'  would process {source.name}')
        return 0

    if not todo:
        print('nothing to do')
        return 0

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ctx = multiprocessing.get_context('spawn')

    started = time.perf_counter()
    games = plies = skipped = failures = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(not args.no_affinity,),
    ) as pool:
        jobs = {
            pool.submit(process_file, source, out_dir, shard_size, args.limit): source
            for source in todo
        }

        for finished, future in enumerate(as_completed(jobs), start=1):
            source = jobs[future]

            try:
                stats = future.result()
            except Exception:
                failures += 1
                print(f'[{finished}/{len(todo)}] {source.name} FAILED', file=sys.stderr)
                traceback.print_exc()
                continue

            games += stats['played']
            plies += stats['plies']
            skipped += stats['skipped']

            rate = stats['played'] / stats['seconds'] if stats['seconds'] else 0
            print(
                f'[{finished}/{len(todo)}] {stats["file"]}: '
                f'{stats["played"]:,} games, {stats["plies"]:,} plies, '
                f'{stats["shards"]} shards, {stats["seconds"] / 3600:.2f} h '
                f'({rate:.0f} games/s, worker {stats["worker"]} core {stats["core"]})'
            )

            if stats['skipped']:
                print(
                    f'    skipped {stats["skipped"]:,} unplayable games, '
                    f'first: {stats["first_error"]}',
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - started

    print()
    print(f'games   : {games:,} ({skipped:,} skipped)')
    print(f'plies   : {plies:,}')
    print(f'failures: {failures} files')
    print(f'wall    : {elapsed / 3600:.2f} h')

    if elapsed:
        print(f'rate    : {games / elapsed:.0f} games/s, {plies / elapsed:,.0f} plies/s')
        print(f'speedup : {plies / elapsed / 6500:.1f}x over one core at 6,500 plies/s')

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
