'''``src/history.py`` - does a window actually hold the plies it claims to?

The window gather in :class:`~src.history.HistoryPositions` is index arithmetic
over a flat array that several games share, and every way it can go wrong is
silent.  A window that runs off the front of one game and into the back of the
one before it still has the right shape, still trains, and teaches the model
that a rook can teleport.

So the checks here are on a synthetic shard whose boards carry their own
identity - square a8 holds the game number, b8 the ply number - which makes
"same game, consecutive plies, newest last" something a test can read straight
off the tensor rather than infer.

The other half is :func:`test_a_later_ply_cannot_change_an_earlier_one`.  The
time axis being causal is what makes every ply in a window a legitimate
training target rather than a leak, and it is one ``is_causal=True`` away from
being silently wrong in the direction that flatters the model: a ply that can
see its own future would predict the hidden board rather well and teach the
network nothing.
'''

import io
import lzma

import numpy as np
import pytest
import torch

from src import preprocess as pp
from src.history import HistoryPositions, encode_histories, flatten_games, histories_loader
from src.regressive import RegressiveConfig, RegressiveFogOfWar

SHARDS = sorted(pp.PROCESSED_DIR.glob(f'*{pp.SHARD_SUFFIX}')) if pp.PROCESSED_DIR.exists() else []
needs_shards = pytest.mark.skipif(not SHARDS, reason=f'no shards under {pp.PROCESSED_DIR}')

HISTORY = 8
GAME, PLY = (0, 0), (0, 1)                                # where identity is written
LENGTHS = [40, 3, 97, 8, 1, 55, 12, 7]                    # deliberately ragged


def labelled_boards(lengths):
    '''``(games, plies, 8, 8)`` int8 boards that say which game and ply they are.'''
    boards = np.zeros((len(lengths), max(lengths), 8, 8), dtype=np.int8)

    for g, length in enumerate(lengths):
        for p in range(length):
            boards[g, p, GAME[0], GAME[1]] = g + 1        # +1: 0 is the padding
            boards[g, p, PLY[0], PLY[1]] = p

    return boards


def as_shard(boards, lengths):
    return {
        'white_board': torch.from_numpy(boards),
        'black_board': torch.from_numpy(boards),
        'correct_board': torch.from_numpy(boards),
        'length': torch.tensor(lengths, dtype=torch.int16),
    }


class Blobs:
    '''The two methods :class:`~src.history.HistoryPositions` asks a cache for.'''

    def __init__(self, shards):
        self.blobs = []
        for boards, lengths in shards:
            buf = io.BytesIO()
            torch.save(as_shard(boards, lengths), buf)
            self.blobs.append(lzma.compress(buf.getvalue(), preset=1))

    def __len__(self):
        return len(self.blobs)

    def read(self, i):
        return self.blobs[i]


@pytest.fixture(scope='module')
def windows():
    '''Every window one synthetic shard yields, as one array.

    ``batch_size=1`` so that the ragged tail the pool drops is empty and the
    test sees every window the dataset can build.
    '''
    dataset = HistoryPositions(
        Blobs([(labelled_boards(LENGTHS), LENGTHS)]), batch_size=1, history=HISTORY,
        pool_shards=1, seed=0, passes=1,
    )

    return torch.cat([view for view, _, _ in dataset]).numpy()


def test_flatten_games_marks_game_starts():
    '''``start`` points at ply 0 of the game each position belongs to.'''
    boards = labelled_boards(LENGTHS)
    view, _, _, start = flatten_games(as_shard(boards, LENGTHS))

    assert len(view) == sum(LENGTHS)                      # padding dropped
    assert view[start, PLY[0], PLY[1]].max() == 0         # every start is a ply 0
    assert np.array_equal(view[:, *GAME], view[start][:, *GAME])


def test_a_window_never_leaves_its_game(windows):
    '''Every ply in a window belongs to the same game as the one it ends on.'''
    games = windows[:, :, GAME[0], GAME[1]]

    assert (games == games[:, -1:]).all()


def test_a_window_is_consecutive_plies_newest_last(windows):
    '''Slot k is exactly ``T - 1 - k`` plies before the one the window ends on.'''
    plies = windows[:, :, PLY[0], PLY[1]]
    back = np.arange(HISTORY - 1, -1, -1)

    assert (plies == plies[:, -1:] - back[None, :]).all()


def test_every_window_is_full(windows):
    '''No padding, no repeats: the earliest slot is a real ply of the game.'''
    plies = windows[:, :, PLY[0], PLY[1]]

    assert (plies >= 0).all()
    assert (plies[:, 0] == plies[:, -1] - (HISTORY - 1)).all()


def test_the_opening_is_covered_by_the_first_full_window(windows):
    '''Plies too early to end a window still appear, as earlier slots.

    This is what makes dropping them as targets harmless: the time axis is
    causal, so ply 2 sitting in slot 2 is predicted from plies 0-2 and nothing
    else - exactly the history it really had.
    '''
    plies = windows[:, :, PLY[0], PLY[1]]

    for early in range(HISTORY - 1):
        assert (plies == early).any(), f'ply {early} never appears in a window'


def test_only_games_too_short_to_fill_a_window_are_dropped(windows):
    '''Every game long enough contributes, and the count is exactly right.'''
    games = set(windows[:, -1, GAME[0], GAME[1]].tolist())
    expected = {g + 1 for g, length in enumerate(LENGTHS) if length >= HISTORY}

    assert games == expected

    wanted = sum(length - HISTORY + 1 for length in LENGTHS if length >= HISTORY)
    assert len(windows) == wanted


def test_a_later_ply_cannot_change_an_earlier_one():
    '''The time axis is causal, so a ply cannot see its own future.

    Without this the older plies would be reading the answer off the newer ones
    and their loss would be meaningless - which is the failure that looks like
    success.
    '''
    torch.manual_seed(0)
    model = RegressiveFogOfWar(RegressiveConfig(d_model=32, n_heads=2, n_layers=2,
                                                d_ff=64, history=4)).eval()

    batch, plies, squares = 4, 4, 64
    piece = torch.randint(0, 7, (batch, plies, squares))
    side = torch.randint(0, 3, (batch, plies, squares))
    visibility = torch.randint(0, 2, (batch, plies, squares))

    with torch.no_grad():
        before = model(piece, side, visibility)

        edited = piece.clone()
        edited[:, -1] = (edited[:, -1] + 3) % 7           # change the newest ply
        after = model(edited, side, visibility)

    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-5)


def test_the_newest_ply_does_read_its_history():
    '''The counterpart: an earlier ply reaches the answer at the newest one.'''
    torch.manual_seed(0)
    model = RegressiveFogOfWar(RegressiveConfig(d_model=32, n_heads=2, n_layers=2,
                                                d_ff=64, history=4)).eval()

    batch, plies, squares = 4, 4, 64
    piece = torch.randint(0, 7, (batch, plies, squares))
    side = torch.randint(0, 3, (batch, plies, squares))
    visibility = torch.randint(0, 2, (batch, plies, squares))

    with torch.no_grad():
        before = model(piece, side, visibility)

        edited = piece.clone()
        edited[:, 0] = (edited[:, 0] + 3) % 7             # change the oldest ply
        after = model(edited, side, visibility)

    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-5)


@needs_shards
def test_encode_histories_describes_every_ply():
    '''``target`` and ``hidden`` cover the window, and ``[-1]`` is the newest ply.'''
    cache = pp.ShardBytes(SHARDS[:1], cache=False)
    loader = histories_loader(pp.ShardSubset(cache, [0]), 256, HISTORY, workers=0,
                              pool_shards=1, seed=0, passes=1, pin_memory=False)

    view, truth, en_passant = next(iter(loader))
    fields = encode_histories(view, truth, en_passant)

    for name in ('piece', 'side', 'visibility', 'target', 'hidden'):
        assert fields[name].shape == (256, HISTORY, 64), name

    alone = pp.encode_batch(truth[:, -1], truth[:, -1], en_passant[:, -1])
    assert torch.equal(fields['target'][:, -1], alone['target'])
    assert torch.equal(fields['hidden'][:, -1], alone['hidden'])


@needs_shards
def test_every_ply_in_a_window_is_a_position_that_was_played():
    '''``truth * mask == view`` across the whole window, not just the newest ply.'''
    cache = pp.ShardBytes(SHARDS[:1], cache=False)
    loader = histories_loader(pp.ShardSubset(cache, [0]), 256, HISTORY, workers=0,
                              pool_shards=1, seed=0, passes=1, pin_memory=False)

    view, truth, en_passant = next(iter(loader))

    wrong = pp.check_consistency(
        view.reshape(-1, 8, 8), truth.reshape(-1, 8, 8), en_passant.reshape(-1)
    )

    assert wrong == 0
