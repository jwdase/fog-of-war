'''``src/preprocess.py`` - is the model being shown the position that was played?

The shards on disk do not carry a visibility mask.  ``src/data.py`` used one to
blank out what white could not see and then threw it away, which leaves a ``0``
meaning two different things: an empty square white is looking at, and a square
white cannot see at all.  ``preprocess.visibility_mask`` rebuilds the mask, and
since the model is handed it as ``e_visibility`` the whole task turns on it
being the same mask ``ChessState`` used.

That is what this file is for.  It is checked two ways, because the two catch
different mistakes:

:func:`test_mask_matches_reference`
    replays real games and compares the rebuilt mask against
    ``ChessState.visible_mask`` square by square.  This is the strict one - it
    catches a mask that is wrong about an *empty* square, which is invisible to
    every other check because an empty square looks the same either way.

:func:`test_mask_agrees_with_the_recorded_view`
    checks ``truth * mask == view`` on shards as they actually are.  Weaker, but
    it runs on millions of positions in a second and it is the invariant
    ``src/train.py`` re-checks at the start of every run, so it is worth knowing
    it holds.
'''

import numpy as np
import pandas as pd
import pytest
import torch

from src import data, game, preprocess as pp
from src.model import (
    CLASS_NAMES,
    FogOfWarNet,
    ModelConfig,
    SIDE_BLACK,
    SIDE_NONE,
    SIDE_WHITE,
    parameter_count,
    probability_of,
    square_index,
)
from main import DATA_DIR


PARQUET = sorted(DATA_DIR.glob('*.parquet'))
SHARDS = sorted(pp.PROCESSED_DIR.glob(f'*{pp.SHARD_SUFFIX}')) if pp.PROCESSED_DIR.exists() else []

needs_games = pytest.mark.skipif(not PARQUET, reason=f'no parquet corpus under {DATA_DIR}')
needs_shards = pytest.mark.skipif(not SHARDS, reason=f'no shards under {pp.PROCESSED_DIR}')


def replay(movetext):
    '''One game as ``(true boards, white's masks)``, straight from ``ChessState``.'''
    state = data.ChessState()
    boards, masks = [], []

    for san in data.san_moves(movetext):
        state.play(san)
        boards.append(state.write_board())
        masks.append(state.visible_mask(game.WHITE).astype(bool))

    return np.stack(boards), np.stack(masks)


@pytest.fixture(scope='module')
def games():
    '''A few dozen real games, as movetext.'''
    return list(pd.read_parquet(PARQUET[0], columns=['moves'])['moves'].head(40))


@pytest.fixture(scope='module')
def shard():
    '''One shard, flattened to the positions it holds.'''
    return pp.flatten_plies(pp.decompress_shard(SHARDS[0].read_bytes()))


# --------------------------------------------------------------------------
# The mask
# --------------------------------------------------------------------------

@needs_games
def test_mask_matches_reference(games):
    '''Ply for ply, square for square, against the generator's own vision.'''
    checked = wrong = 0

    for movetext in games:
        boards, reference = replay(movetext)
        en_passant = pp.ep_columns(boards[None])[0]

        rebuilt = pp.visibility_mask(
            torch.from_numpy(boards), torch.from_numpy(en_passant)
        ).numpy()

        wrong += int((rebuilt != reference).sum())
        checked += reference.size

    assert checked > 100_000, 'not enough positions to mean anything'
    assert wrong == 0, f'{wrong} of {checked} squares disagree with ChessState'


@needs_games
def test_en_passant_is_worth_having(games):
    '''The en passant square is rare, real, and the only thing a board cannot say.

    If this ever stops failing without ``en_passant``, the handling in
    :func:`~src.preprocess.visibility_mask` has become dead code and should go.
    '''
    missed = plies = 0

    for movetext in games:
        boards, reference = replay(movetext)
        en_passant = pp.ep_columns(boards[None])[0]
        plies += int((en_passant >= 0).sum())

        blind = pp.visibility_mask(torch.from_numpy(boards), None).numpy()
        missed += int((blind != reference).sum())

    assert plies > 0, 'no double pushes in the sample - pick a bigger one'
    assert missed > 0, 'ignoring en passant cost nothing; the special case is dead'


@needs_games
def test_a_pawn_sees_two_squares_only_from_home(games):
    '''The bug this file was written for.

    A pawn on its home rank sees two squares ahead; a pawn that has already
    moved sees one.  Getting that wrong is invisible in the ``truth * mask``
    invariant, because the extra square it wrongly reveals is an empty one.
    '''
    board = np.zeros((1, 8, 8), dtype=np.int8)
    board[0, 6, 4] = -game.PAWN                      # a white pawn on e2

    home = pp.visibility_mask(torch.from_numpy(board))[0]
    assert bool(home[5, 4]) and bool(home[4, 4]), 'e3 and e4 are both in view from e2'
    assert not bool(home[3, 4]), 'e5 is not'

    board[0, 6, 4], board[0, 4, 4] = 0, -game.PAWN   # the same pawn, now on e4
    moved = pp.visibility_mask(torch.from_numpy(board))[0]
    assert bool(moved[3, 4]), 'e5 is in view from e4'
    assert not bool(moved[2, 4]), 'e6 is not - only a pawn at home pushes twice'


def test_a_ray_stops_at_the_first_piece():
    '''What blocks a rook is seen; what stands behind it is not.'''
    board = np.zeros((1, 8, 8), dtype=np.int8)
    board[0, 7, 0] = -game.ROOK                      # white rook a1
    board[0, 4, 0] = game.KNIGHT                     # black knight a4
    board[0, 2, 0] = game.QUEEN                      # black queen a6, behind it

    seen = pp.visibility_mask(torch.from_numpy(board))[0]

    assert bool(seen[4, 0]), 'the blocking knight is what the rook can see'
    assert not bool(seen[3, 0]) and not bool(seen[2, 0]), 'nothing past it is'


@needs_shards
def test_mask_agrees_with_the_recorded_view(shard):
    '''``truth * mask == view``, on a whole shard of real positions.'''
    view, truth, en_passant = (torch.from_numpy(part) for part in shard)

    assert len(view) > 50_000
    assert pp.check_consistency(view, truth, en_passant) == 0


# --------------------------------------------------------------------------
# En passant detection
# --------------------------------------------------------------------------

@needs_games
def test_ep_columns_finds_black_double_pushes(games):
    '''Every file it flags really does hold a black pawn that just came two.'''
    for movetext in games:
        boards, _ = replay(movetext)
        columns = pp.ep_columns(boards[None])[0]

        for ply in np.flatnonzero(columns >= 0):
            column = columns[ply]

            assert ply % 2 == 1, 'only black can have just moved on an odd ply'
            assert boards[ply][3, column] == game.PAWN
            assert boards[ply - 1][1, column] == game.PAWN
            assert boards[ply][1, column] == game.EMPTY_SQUARE


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

@needs_shards
def test_encode_batch_says_what_is_on_the_board(shard):
    '''The four encoders against the boards they were built from.'''
    view, truth, en_passant = (torch.from_numpy(part[:4096]) for part in shard)
    fields = pp.encode_batch(view, truth, en_passant)

    flat_view = view.reshape(-1, 64)
    flat_truth = truth.reshape(-1, 64)

    assert torch.equal(fields['piece'], flat_view.abs().long())

    # src/game.py writes white negative on the board; e_side writes it +1.
    assert torch.equal(fields['side'] == SIDE_WHITE, flat_view < 0)
    assert torch.equal(fields['side'] == SIDE_BLACK, flat_view > 0)
    assert torch.equal(fields['side'] == SIDE_NONE, flat_view == 0)

    assert torch.equal(fields['hidden'].reshape(-1, 64), fields['visibility'] == 0)

    # A visible square is what is there; a hidden one reads empty in the view
    # and keeps its real class in the target, which is the whole task.
    visible = ~fields['hidden'].reshape(-1, 64)
    assert torch.equal(flat_view[visible], flat_truth[visible])
    assert (flat_view[~visible] == 0).all()

    white_pawn = CLASS_NAMES.index('P')
    black_queen = CLASS_NAMES.index('q')
    assert torch.equal(fields['target'] == white_pawn, flat_truth == -game.PAWN)
    assert torch.equal(fields['target'] == black_queen, flat_truth == game.QUEEN)


def test_white_sees_its_own_half_at_the_start():
    '''Ranks 1 to 4 and not a square past them.

    Rank 3 because every pawn can push to it, rank 4 because every pawn can
    still push twice, and nothing beyond because the fog starts there.
    '''
    start = torch.from_numpy(data.ChessState().write_board()).unsqueeze(0)
    seen = pp.visibility_mask(start)[0]

    assert seen[4:].all(), 'white can see its own two ranks and the two above'
    assert not seen[:4].any(), 'and nothing on the far half of the board'


def test_a_pawn_leaves_a_hole_behind_it():
    '''After 1. e4, white cannot see e3.

    Which is the fog being genuinely strange, and worth pinning down: the pawn
    that used to cover e3 by standing next to it is now two squares past it, and
    nothing else on white's first two ranks reaches that square.  A model that
    only ever sees the mask as "white's half" would be learning the wrong thing.
    '''
    state = data.ChessState()
    state.play('e4')

    seen = pp.visibility_mask(torch.from_numpy(state.write_board()).unsqueeze(0))[0]
    reference = state.visible_mask(game.WHITE).astype(bool)

    assert not seen[5, 4], 'e3 is behind the pawn and nothing else covers it'
    assert bool(seen[3, 4]), 'e5 is in front of it and is'
    assert (seen.numpy() == reference).all()


@needs_shards
def test_a_shard_position_is_already_foggy(shard):
    '''The very first ply on disk already hides most of the board.'''
    view, truth, en_passant = (torch.from_numpy(part[:1]) for part in shard)
    fields = pp.encode_batch(view, truth, en_passant)

    hidden = fields['hidden'].reshape(8, 8)
    assert hidden[0].all(), "none of black's back rank is visible"
    assert 20 <= int(hidden.sum()) <= 40, 'about half the board is fog'


def test_flatten_drops_the_padding():
    '''Only the plies a game really has come out of a shard.'''
    lengths = torch.tensor([3, 1, 2], dtype=torch.int16)
    boards = torch.zeros((3, 4, 8, 8), dtype=torch.int8)

    for row, length in enumerate(lengths):
        boards[row, :length] = row + 1

    view, truth, en_passant = pp.flatten_plies({
        'white_board': boards, 'correct_board': boards, 'length': lengths,
    })

    assert len(view) == 6 == len(truth) == len(en_passant)
    assert [int(board[0, 0]) for board in truth] == [1, 1, 1, 2, 3, 3]


# --------------------------------------------------------------------------
# The model's end of the contract
# --------------------------------------------------------------------------

def test_the_model_is_the_size_it_is_budgeted_for():
    '''~50,000 parameters, and a change to that should be a deliberate one.'''
    assert parameter_count(FogOfWarNet(ModelConfig())) == 49_205


def test_squares_are_named_the_way_people_name_them():
    assert square_index('a8') == 0
    assert square_index('a7') == 8
    assert square_index('h1') == 63

    # Which has to be how src/game.py lays a board out, since that is where the
    # tensors come from: row 0 is rank 8, column 0 is file a.
    board = np.arange(64, dtype=np.int8).reshape(8, 8)
    assert board[1, 0] == square_index('a7')


def test_a_query_reads_out_of_the_right_place():
    '''``P(black pawn on a7)`` is one number, and it is the one asked for.'''
    model = FogOfWarNet()
    batch = {name: torch.zeros(2, 64, dtype=torch.long) for name in
             ('piece', 'side', 'visibility')}

    probabilities = model.predict(**batch)

    assert probabilities.shape == (2, 64, len(CLASS_NAMES))
    assert torch.allclose(probabilities.sum(-1), torch.ones(2, 64), atol=1e-5)

    asked = probability_of(probabilities, 'a7', game.PAWN, -1)
    assert torch.equal(asked, probabilities[:, square_index('a7'), CLASS_NAMES.index('p')])
