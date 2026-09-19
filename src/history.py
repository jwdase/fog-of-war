'''Windows of consecutive plies, for :class:`~src.regressive.RegressiveFogOfWar`.

``src/preprocess.py`` flattens a shard into independent positions and forgets
which game each came from, which is all :class:`~src.model.FogOfWarNet` needs.
The history model needs the opposite: position ``i`` together with the ``T-1``
plies in front of it, and a promise that they belong to the same game.

Both facts are already in the shard - ``src/data.py`` writes one row per game,
``(games, plies, 8, 8)``, with a ``length`` saying where each game really ends -
so nothing has to be reprocessed.  What this module adds is the bookkeeping.

Why windows are gathered and not stored
---------------------------------------

The obvious implementation materialises ``(positions, T, 8, 8)`` and hands out
slices.  At ``T=8`` that is eight copies of the corpus: a pool of four shards
goes from ~21 MB to ~170 MB per loader worker, times twenty-four workers, for
data that is 87% duplicated - every ply appears in eight windows.

So the pool stays flat, exactly as ``FogPositions`` leaves it, and a batch is
gathered by *index*: ``B`` chosen positions become a ``(B, T)`` array of row
numbers, and one fancy-index produces the window.  At ``B=1024, T=8`` that is
8,192 rows of 64 bytes - half a megabyte - built per batch and thrown away.

The start of a game
-------------------

Position 3 of a game has no eighth-previous ply, so it cannot end a full
window.  Those plies are not lost: the model's time axis is causal and every
ply of a window is a training target, so ply 3 is supervised as the fourth slot
of the window ending at ply 7 - conditioned on plies 0 to 3 and nothing else,
which is exactly its own history.  Only whole games shorter than the window are
dropped, and at ``T=8`` that is 0.5% of them.

This is why there is no padding and no mask here.  Every slot of every window
is a real ply of a real game, which is worth more than it sounds: a padded slot
would need a mask through the attention, and an explicit mask is what knocks
``scaled_dot_product_attention`` off its flash kernel - it asked for 16 GiB of
backward workspace before this was arranged away.
'''

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.model import N_SQUARES
from src.preprocess import (
    ShardSubset,                                          # noqa: F401 - re-exported
    _worker_setup,
    decompress_shard,
    encode_batch,
    ep_columns,
)


def flatten_games(shard):
    '''One shard flat, plus where each position's game starts.

    ``(view, truth, en_passant, start)``: the first three are exactly what
    :func:`~src.preprocess.flatten_plies` returns, and ``start`` is a
    ``(positions,)`` index saying which row is ply 0 of the game this position
    belongs to.  Games are contiguous and in order in the flat array - the mask
    is applied row-major over ``(games, plies)`` - so a window is a slice and
    "same game" is an inequality rather than a lookup.
    '''
    view = shard['white_board'].numpy()
    truth = shard['correct_board'].numpy()
    lengths = shard['length'].numpy().astype(np.int64)

    en_passant = ep_columns(truth)
    real = np.arange(truth.shape[1])[None, :] < lengths[:, None]

    first = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    return view[real], truth[real], en_passant[real], np.repeat(first, lengths)


class HistoryPositions(IterableDataset):
    '''Shuffled batches of ``T``-ply windows, streamed from shards.

    The pooling and shuffling are :class:`~src.preprocess.FogPositions`' - the
    positions a batch is drawn from come from four shards at once so that a
    batch is not forty consecutive plies of the same game - with the window
    gather on top.  Note that shuffling stays over *target* plies: the windows
    two neighbouring targets pull in overlap by seven eighths, but they are
    never in the same batch, which is what the shuffle is for.
    '''

    def __init__(self, shards, batch_size, history, pool_shards=4, seed=0, passes=None):
        if history < 1:
            raise ValueError(f'a window is at least one ply, got {history}')

        self.shards = shards
        self.batch_size = batch_size
        self.history = history
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
        parts = [flatten_games(decompress_shard(self.shards.read(i))) for i in shard_ids]

        view = np.concatenate([part[0] for part in parts])
        truth = np.concatenate([part[1] for part in parts])
        en_passant = np.concatenate([part[2] for part in parts])

        # Each part's start indices are relative to its own shard; concatenating
        # the positions shifts them by everything already in the pool.
        sizes = np.array([len(part[0]) for part in parts])
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        start = np.concatenate([part[3] + off for part, off in zip(parts, offsets)])

        # (T,) of how far back each slot reaches; the last slot ends the window.
        back = np.arange(self.history - 1, -1, -1)

        # A position can end a window only if its game reaches back far enough.
        # Every earlier ply still gets trained on, as an earlier slot of the
        # first window its game can fill - see the module docstring.
        depth = np.arange(len(view)) - start
        eligible = np.flatnonzero(depth >= self.history - 1)

        shuffled = rng.permutation(eligible)

        for at in range(0, len(shuffled) - self.batch_size + 1, self.batch_size):
            take = shuffled[at:at + self.batch_size]
            rows = take[:, None] - back[None, :]

            yield (
                torch.from_numpy(view[rows]),
                torch.from_numpy(truth[rows]),
                torch.from_numpy(en_passant[rows]),
            )


def histories_loader(shards, batch_size, history, workers, pool_shards=4, seed=0,
                     passes=None, prefetch=4, pin_memory=True):
    '''A DataLoader over :class:`HistoryPositions` with the settings it needs.'''
    dataset = HistoryPositions(shards, batch_size, history, pool_shards, seed, passes)

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0 and passes is None,
        prefetch_factor=prefetch if workers > 0 else None,
        worker_init_fn=_worker_setup if workers > 0 else None,
    )


def encode_histories(view, truth, en_passant):
    '''``(batch, T, ...)`` boards as the tensors the history model eats.

    Every ply in the window is encoded the same way a lone position is - the
    window is folded into the batch axis and handed to the tested
    :func:`~src.preprocess.encode_batch`, so the visibility ray-march here is
    the one ``tests/test_preprocess.py`` checks square for square.

    Everything comes back ``(batch, T, 64)``, ``target`` and ``hidden``
    included: the time axis is causal, so every ply is predicted from its own
    past and every ply is a target.  The model never sees the truth of any ply,
    including the ones it has already been shown - only the view, and the
    visibility mask derived from the truth, which is what white knew it could
    see at the time.

    ``[:, -1]`` is the deployment query, and what ``src/train_regressive.py``
    validates on.
    '''
    batch, plies = view.shape[:2]

    fields = encode_batch(
        view.reshape(batch * plies, 8, 8),
        truth.reshape(batch * plies, 8, 8),
        en_passant.reshape(batch * plies),
    )

    spread = lambda name: fields[name].view(batch, plies, N_SQUARES)

    return {name: spread(name)
            for name in ('piece', 'side', 'visibility', 'target', 'hidden')}
