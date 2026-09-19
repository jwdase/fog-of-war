'''Shards on disk in, batches of model inputs out.

``src/data.py`` wrote the corpus as ~26,000 compressed shards of 1000 games
each, every game a row of ``(plies, 8, 8)`` int8 boards::

    white_board     what white could see after each ply
    black_board     what black could see           (not used here - white only)
    correct_board   what was actually there
    length          where each game really ends, the rest being padding

Three things have to happen between that and :class:`src.model.FogOfWarNet`, and
they are split by where they are cheapest to do:

**On the worker processes** - :func:`flatten_plies` throws away the padding and
the black point of view and hands back a flat run of positions, plus the one
piece of state the boards do not carry: whether black has just double-pushed a
pawn, from :func:`ep_columns`.  Decompressing a shard is ~0.35 s and is the only
real CPU cost in the pipeline, so it wants as many workers as the node has.

**On the GPU** - :func:`encode_batch` turns a pair of int8 boards into the four
index tensors the model eats.  The expensive part of that is
:func:`visibility_mask`, which is a ray-march over the board, and it is perfectly
parallel across the batch.

**In RAM** - :class:`ShardBytes` holds the whole compressed corpus, ~21 GB, so
the shared filesystem is read once for the entire run rather than once per pass.

The reconstructed visibility mask
---------------------------------

``src/data.py`` saved what white could *see* but not *where it could see*, and
those are different: a square reads 0 both when it is visibly empty and when it
is hidden.  :func:`visibility_mask` rebuilds the mask that
``ChessState.visible_mask`` would have produced, by re-deriving white's
pseudo-legal moves from the true board - all 64 squares at once, in tensor ops,
instead of one python-chess board per piece.

It is built from ``correct_board`` rather than ``white_board`` because those two
disagree in one case: a hidden piece standing directly in front of a white pawn
blocks the push, so the square is *not* visible, and white's view alone cannot
tell that square from an empty one.  Deriving the mask from the truth reproduces
the generator exactly, and ``tests/test_preprocess.py`` checks it ply by ply
against ``ChessState`` on real games.

This is not a leak.  The mask is not extra information about the position - it
is the shape of white's own ignorance, which a real player computing its own
legal moves knows for free.  What it must never do is disagree with the
observation it is paired with, and ``correct_board * mask == white_board`` is
the invariant that says it does not; :func:`check_consistency` asserts it.
'''

from __future__ import annotations

import io
import lzma
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src import game
from src.model import (
    HIDDEN,
    N_SQUARES,
    SIDE_BLACK,
    SIDE_NONE,
    SIDE_WHITE,
    VISIBLE,
)


#: Where ``scripts/process_corpus.sbatch`` put the shards.
PROCESSED_DIR = Path(
    os.environ.get('FOW_PROCESSED', '/home/jwdase/orcd/pool/fog-of-war/processed')
)

#: What ``src/data.py`` calls a shard.  Kept in step with ``data.SHARD_SUFFIX``.
SHARD_SUFFIX = '.pt.xz'


# --------------------------------------------------------------------------
# Finding and reading shards
# --------------------------------------------------------------------------

def list_shards(directory=PROCESSED_DIR):
    '''Every shard in ``directory``, in a stable order.

    Sorted, because the train/validation split is a function of position in this
    list and has to survive being recomputed on another node.
    '''
    shards = sorted(Path(directory).glob(f'*{SHARD_SUFFIX}'))

    if not shards:
        raise FileNotFoundError(f'no {SHARD_SUFFIX} shards under {directory}')

    return shards


def split_shards(shards, n_validation, seed=0):
    '''``(train, validation)`` - a fixed random slice of whole shards held out.

    Whole shards rather than a fraction of every shard, because a shard is 1000
    whole games: splitting inside one would put the same game's early plies in
    training and its late plies in validation, which are not independent.
    '''
    order = np.random.default_rng(seed).permutation(len(shards))
    held_out = set(order[:n_validation].tolist())

    return (
        [shard for i, shard in enumerate(shards) if i not in held_out],
        [shard for i, shard in enumerate(shards) if i in held_out],
    )


def decompress_shard(raw):
    '''The ``{white_board, black_board, correct_board, length}`` dict from bytes.

    The in-memory counterpart of ``src.data.load_shard``, which takes a path.
    Both have to agree with ``src.data.save_shard``, which is xz around a
    ``torch.save``.
    '''
    return torch.load(io.BytesIO(lzma.decompress(raw)), weights_only=True)


class ShardBytes:
    '''The compressed corpus, held in RAM.

    ~21 GB for the whole thing, against 256 GB on the training node, so the
    shared filesystem is read once at startup instead of on every pass.  That
    matters twice over: a 12-hour run makes dozens of passes, and the corpus
    lives on a parallel filesystem other people are also using.

    Everything is read into one flat ``uint8`` array before the DataLoader forks
    its workers, so the workers inherit it copy-on-write and no shard is ever
    copied per worker - only the ~780 kB slice each one is decompressing.

    ``cache=False`` falls back to reading each shard from disk as it is wanted,
    which is what a smoke test on a login node wants.
    '''

    def __init__(self, paths, cache=True, threads=16, log=print):
        self.paths = list(paths)
        self.buffer = None
        self.offsets = None

        if cache:
            self._load(threads, log)

    def _load(self, threads, log):
        sizes = np.array([path.stat().st_size for path in self.paths], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])

        total = int(self.offsets[-1])
        log(f'caching {len(self.paths):,} shards ({total / 2**30:.1f} GiB) in RAM')
        self.buffer = np.empty(total, dtype=np.uint8)

        def read(i):
            with open(self.paths[i], 'rb') as handle:
                handle.readinto(
                    memoryview(self.buffer)[self.offsets[i]:self.offsets[i + 1]]
                )

        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(read, range(len(self.paths))):
                pass

        log(f'cached {total / 2**30:.1f} GiB')

    @property
    def cached(self):
        return self.buffer is not None

    def __len__(self):
        return len(self.paths)

    def read(self, i):
        '''Shard ``i`` as the compressed bytes ``decompress_shard`` wants.'''
        if self.buffer is None:
            return self.paths[i].read_bytes()

        return self.buffer[self.offsets[i]:self.offsets[i + 1]].tobytes()


# --------------------------------------------------------------------------
# Shard to positions
# --------------------------------------------------------------------------

def ep_columns(true_boards):
    '''``(games, plies)`` int8: the file black has just double-pushed on, else -1.

    The one thing a board array cannot say and white's vision depends on.  A
    white pawn on the fifth rank sees the square *behind* an adjacent pawn that
    has just come past it - the en passant capture is a pseudo-legal move, so
    ``ChessState`` counts that square as seen - and nothing in the position
    itself records that the push was the move just played.

    It can be read off two consecutive boards instead: a black pawn that was on
    rank 7 and is now on rank 5 with rank 7 vacated has just double-pushed.
    Only on black's plies, which are the odd ones, since white moves first.
    '''
    n_games, n_plies = true_boards.shape[:2]
    columns = np.full((n_games, n_plies), -1, dtype=np.int8)

    if n_plies < 2:
        return columns

    before, after = true_boards[:, :-1], true_boards[:, 1:]

    pushed = (
        (after[:, :, 3, :] == game.PAWN)
        & (before[:, :, 1, :] == game.PAWN)
        & (after[:, :, 1, :] == game.EMPTY_SQUARE)
        & (before[:, :, 3, :] == game.EMPTY_SQUARE)
    )
    pushed &= (np.arange(1, n_plies) % 2 == 1)[None, :, None]

    columns[:, 1:] = np.where(pushed.any(2), pushed.argmax(2), -1)

    return columns


def flatten_plies(shard):
    '''One shard as a flat run of positions: ``(view, truth, en passant file)``.

    Drops the padding past each game's last ply and black's point of view, which
    together are most of what was loaded: a shard is 1000 games padded out to
    the longest of them, ~188 plies, and they average ~83.
    '''
    view = shard['white_board'].numpy()
    truth = shard['correct_board'].numpy()
    lengths = shard['length'].numpy().astype(np.int64)

    en_passant = ep_columns(truth)
    real = np.arange(truth.shape[1])[None, :] < lengths[:, None]

    return view[real], truth[real], en_passant[real]


class FogPositions(IterableDataset):
    '''Shuffled batches of white-to-view positions, streamed from shards.

    Batches are assembled here rather than by a collate function because the
    unit of work is a shard, not a position: decompressing one yields ~83,000
    positions at once, and handing them out one at a time only to have them
    stacked again is pure overhead.  The DataLoader is therefore built with
    ``batch_size=None``.

    Positions from one game are highly correlated - a board barely changes from
    ply to ply - so they are mixed before they are handed out: ``pool_shards``
    shards are decompressed together, concatenated and permuted, which puts
    ~330,000 positions from 4000 different games in the pool a batch is drawn
    from.  Shard *order* is reshuffled on every pass as well, so the pools
    differ from one pass to the next.

    ``passes=None`` streams for ever, which is what training wants when the stop
    condition is a wall clock.  A finite number of passes is what validation
    wants, and with ``seed`` fixed it reads the same positions every time.
    '''

    def __init__(self, shards, batch_size, pool_shards=4, seed=0, passes=None):
        self.shards = shards
        self.batch_size = batch_size
        self.pool_shards = pool_shards
        self.seed = seed
        self.passes = passes

    def __iter__(self):
        info = get_worker_info()
        worker, n_workers = (0, 1) if info is None else (info.id, info.num_workers)

        rng = np.random.default_rng([self.seed, worker])
        order = np.arange(len(self.shards))

        pass_number = 0
        while self.passes is None or pass_number < self.passes:
            rng.shuffle(order)
            mine = order[worker::n_workers]

            for start in range(0, len(mine), self.pool_shards):
                yield from self._pool(mine[start:start + self.pool_shards], rng)

            pass_number += 1

    def _pool(self, shard_ids, rng):
        '''Every batch that comes out of one pool of shards, in random order.'''
        parts = [flatten_plies(decompress_shard(self.shards.read(i))) for i in shard_ids]

        view = np.concatenate([part[0] for part in parts])
        truth = np.concatenate([part[1] for part in parts])
        en_passant = np.concatenate([part[2] for part in parts])

        shuffled = rng.permutation(len(view))

        # A short tail is dropped rather than padded: the pool holds hundreds of
        # thousands of positions, so this is a rounding error, and a ragged
        # batch would cost a recompilation under torch.compile.
        for start in range(0, len(shuffled) - self.batch_size + 1, self.batch_size):
            take = shuffled[start:start + self.batch_size]

            yield (
                torch.from_numpy(view[take]),
                torch.from_numpy(truth[take]),
                torch.from_numpy(en_passant[take]),
            )


class ShardSubset:
    '''One split's shards, read through a :class:`ShardBytes` shared with the rest.

    Training and validation are disjoint sets of shards out of one cache, so the
    corpus is held in RAM once rather than once per split.  A module-level class
    rather than a closure so that it survives being handed to a worker process.
    '''

    def __init__(self, cache, ids):
        self.cache = cache
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def read(self, i):
        return self.cache.read(self.ids[i])


def _worker_setup(_):
    '''One thread per worker: 32 of them fighting over BLAS pools helps nobody.'''
    torch.set_num_threads(1)


def positions_loader(shards, batch_size, workers, pool_shards=4, seed=0, passes=None,
                     prefetch=4, pin_memory=True):
    '''A DataLoader over :class:`FogPositions` with the settings it needs.'''
    dataset = FogPositions(shards, batch_size, pool_shards, seed, passes)

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0 and passes is None,
        prefetch_factor=prefetch if workers > 0 else None,
        worker_init_fn=_worker_setup if workers > 0 else None,
    )


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------

#: Where a rook and a bishop walk, as ``(row, column)`` steps.  Row 0 is rank 8,
#: so white advances by -1.
ROOK_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
KING_STEPS = ROOK_STEPS + BISHOP_STEPS
KNIGHT_STEPS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)


def _shift(mask, d_row, d_column):
    '''``mask`` slid by one step, with whatever falls off the board dropped.'''
    moved = torch.zeros_like(mask)

    row_from = slice(max(0, -d_row), 8 - max(0, d_row))
    column_from = slice(max(0, -d_column), 8 - max(0, d_column))
    row_to = slice(row_from.start + d_row, row_from.stop + d_row)
    column_to = slice(column_from.start + d_column, column_from.stop + d_column)

    moved[..., row_to, column_to] = mask[..., row_from, column_from]

    return moved


def _rays(sliders, free, steps):
    '''Every square a slider on ``sliders`` can walk to along ``steps``.

    The ray stops *after* the first piece it meets rather than before it: the
    blocker is what is seen.  Behind it is the fog.
    '''
    seen = torch.zeros_like(sliders)

    for d_row, d_column in steps:
        reach = sliders

        for _ in range(7):
            reach = _shift(reach, d_row, d_column)
            seen |= reach
            reach = reach & free

    return seen


def visibility_mask(true_board, en_passant=None):
    '''``(batch, 8, 8)`` bool: where white can see, given the true position.

    A reimplementation of ``ChessState.visible_squares(WHITE)`` in tensor ops.
    White sees its own pieces and everywhere they could move to - pseudo-legally,
    so pins do not narrow the view - except the king, which sees the eight
    squares around it whether or not stepping there would be suicide.

    Squares holding white's own pieces come back visible even though a piece
    cannot move onto them, because the generator adds every piece's own square
    before it asks where the piece could go.

    ``en_passant`` is the ``(batch,)`` file from :func:`ep_columns`, or ``None``
    to ignore the case entirely; it is worth about one square in a thousand
    positions and is the only part of white's vision the board alone cannot say.
    '''
    board = true_board.reshape(-1, 8, 8)

    # src/game.py writes white negative.
    white = board < 0
    enemy = board > 0
    free = board == 0

    pawns = board == -game.PAWN
    knights = board == -game.KNIGHT
    kings = board == -game.KING
    straight = (board == -game.ROOK) | (board == -game.QUEEN)
    diagonal = (board == -game.BISHOP) | (board == -game.QUEEN)

    seen = white.clone()
    seen |= _rays(straight, free, ROOK_STEPS)
    seen |= _rays(diagonal, free, BISHOP_STEPS)

    for step in KNIGHT_STEPS:
        seen |= _shift(knights, *step)

    for step in KING_STEPS:
        seen |= _shift(kings, *step)

    # A pawn sees the square it can push to, which it does not attack, and the
    # diagonals only when there is something there to take.
    push = _shift(pawns, -1, 0) & free
    seen |= push

    # The second step of a double push, which only a pawn still on its home
    # rank has: row 6 to row 4, so ``push`` has to have reached row 5 and row 4
    # has to be free as well.  No other row is a legal landing square, which is
    # what keeps a pawn already on the march from seeing two squares ahead.
    seen[:, 4, :] |= push[:, 5, :] & free[:, 4, :]

    seen |= _shift(pawns, -1, -1) & enemy
    seen |= _shift(pawns, -1, 1) & enemy

    if en_passant is not None:
        seen |= _en_passant_square(pawns, en_passant)

    return seen


def _en_passant_square(white_pawns, en_passant):
    '''The square behind a black pawn that has just run past a white one.

    Black's pawn is on row 3 - the fifth rank - and the square it skipped is row
    2.  White may capture onto it from row 3 either side, and a pseudo-legal
    move is a seen square.
    '''
    files = torch.arange(8, device=en_passant.device)
    en_passant = en_passant.reshape(-1, 1).long()

    on_fifth = white_pawns[:, 3, :]
    beside = torch.zeros_like(on_fifth)
    beside[:, :7] |= on_fifth[:, 1:]                       # a white pawn one file right
    beside[:, 1:] |= on_fifth[:, :7]                       # one file left

    seen = torch.zeros_like(white_pawns)
    seen[:, 2, :] = beside & (files == en_passant) & (en_passant >= 0)

    return seen


# --------------------------------------------------------------------------
# Positions to model inputs
# --------------------------------------------------------------------------

#: Signed piece code + 6 -> output class.  White is negative on the board and
#: takes classes 1-6; black is positive and takes 7-12.
_CLASS_FROM_CODE = torch.tensor(
    [6, 5, 4, 3, 2, 1, 0, 7, 8, 9, 10, 11, 12], dtype=torch.long
)

#: Signed piece code + 6 -> side index.  Same table, for ``e_side``.
_SIDE_FROM_CODE = torch.tensor(
    [SIDE_WHITE] * 6 + [SIDE_NONE] + [SIDE_BLACK] * 6, dtype=torch.long
)


_TABLE_CACHE = {}


def _on(table, device):
    '''``table`` on ``device``, built once rather than copied over every step.'''
    key = (id(table), str(device))

    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = table.to(device)

    return _TABLE_CACHE[key]


def encode_batch(view, truth, en_passant):
    '''One batch of boards as the tensors :class:`~src.model.FogOfWarNet` eats.

    Returns ``piece``, ``side`` and ``visibility`` - the model's three
    ``(batch, 64)`` inputs - together with the ``target`` class per square and
    the ``hidden`` mask the interesting metrics are computed over.

    Takes the int8 boards already on the device they are wanted on - the work is
    indexing and a ray-march, both far cheaper on a GPU than on a worker
    process, and int8 is eight times less to send across the bus than the int64
    the embeddings need.  Nothing here moves a tensor between devices, so the
    whole function is one ``torch.compile`` region.
    '''
    classes = _on(_CLASS_FROM_CODE, truth.device)
    sides = _on(_SIDE_FROM_CODE, truth.device)

    codes = view.reshape(-1, N_SQUARES).long()
    visible = visibility_mask(truth, en_passant).reshape(-1, N_SQUARES)

    return {
        'piece': codes.abs(),
        'side': sides[codes + 6],
        'visibility': torch.where(visible, VISIBLE, HIDDEN),
        'target': classes[truth.reshape(-1, N_SQUARES).long() + 6],
        'hidden': ~visible,
    }


def check_consistency(view, truth, en_passant):
    '''The invariant tying a rebuilt mask to the view it is paired with.

    ``ChessState`` wrote ``white_board`` as the true board with everything white
    cannot see blanked out, so a correct mask satisfies ``truth * mask == view``
    exactly.  Returns the number of squares where it does not.
    '''
    mask = visibility_mask(truth, en_passant)
    blanked = torch.where(mask, truth.reshape(-1, 8, 8), torch.zeros_like(truth))

    return int((blanked != view.reshape(-1, 8, 8)).sum())
