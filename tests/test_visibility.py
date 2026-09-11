'''The fog-of-war board state: what one side can see of ``ChessState.data``.

``visible_positions`` projects the full board down to the squares a side's own
pieces stand on, plus every square those pieces could move to. Anything else
comes back as ``EMPTY_SQUARE``.

``gen_state`` runs that for both sides. It cannot run today: it hands
``visible_positions`` a piece dictionary containing ``KING``, and ``KING`` has no
entry in ``MOVE_FUNCTIONS``, so the lookup raises ``KeyError(6)``. The tests
below work around that by passing king-less dictionaries, which exercises the
same code path; the ones that need the whole dictionary are marked xfail.
'''

import numpy as np
import pytest

from src.data.data import (
    BISHOP,
    COORDINATES,
    EMPTY_SQUARE,
    KING,
    KNIGHT,
    MOVE_FUNCTIONS,
    PAWN,
    QUEEN,
    ROOK,
)

from .helpers import occupied_squares, piece_at, square

NO_KING_IN_REGISTRY = "visible_positions raises KeyError(6): KING is not in MOVE_FUNCTIONS"


def without_king(pieces):
    '''A side's pieces minus the king, which has no generator to look up.'''
    return {piece: list(squares) for piece, squares in pieces.items() if piece != KING}


def seen(positions):
    '''The squares a view reports something on.'''
    return {
        square(row, col)
        for row in range(8)
        for col in range(8)
        if positions[row, col] != EMPTY_SQUARE
    }


def reachable(pieces):
    '''Own squares plus every square those pieces could move to.'''
    squares = set()
    for piece, locations in pieces.items():
        for location in locations:
            squares.add(location)
            squares.update(MOVE_FUNCTIONS[piece](location))
    return squares


@pytest.fixture
def white(state):
    return without_king(state.white_pieces)


@pytest.fixture
def black(state):
    return without_king(state.black_pieces)


# --------------------------------------------------------------------------- #
# Shape of the view
# --------------------------------------------------------------------------- #

def test_a_view_is_an_eight_by_eight_integer_board(state, white):
    positions = state.visible_positions(white)
    assert positions.shape == (8, 8)
    assert np.issubdtype(positions.dtype, np.integer)


def test_a_side_with_no_pieces_sees_nothing(state):
    assert not state.visible_positions({}).any()


def test_a_view_is_a_fresh_array_each_time(state, white):
    first = state.visible_positions(white)
    second = state.visible_positions(white)
    assert first is not second
    assert np.array_equal(first, second)


def test_building_a_view_does_not_disturb_the_board(state, white):
    before = state.data.copy()
    state.visible_positions(white)
    assert np.array_equal(state.data, before)


def test_building_a_view_does_not_disturb_the_pieces(state):
    before = {piece: list(squares) for piece, squares in state.white_pieces.items()}
    state.visible_positions(without_king(state.white_pieces))
    assert state.white_pieces == before


# --------------------------------------------------------------------------- #
# What a view is allowed to contain
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("side", ["white", "black"])
def test_a_view_never_invents_a_piece(state, side, request):
    '''Every square the view reports holds exactly what the real board holds.'''
    positions = state.visible_positions(request.getfixturevalue(side))
    for coord in seen(positions):
        assert piece_at(positions, coord) == piece_at(state.data, coord)


@pytest.mark.parametrize("side", ["white", "black"])
def test_your_own_pieces_are_always_visible(state, side, request):
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces)
    for piece, locations in pieces.items():
        for location in locations:
            assert piece_at(positions, location) == piece


@pytest.mark.parametrize("side", ["white", "black"])
def test_nothing_outside_reach_is_visible(state, side, request):
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces)
    assert seen(positions) <= reachable(pieces)


@pytest.mark.parametrize("side", ["white", "black"])
def test_everything_within_reach_is_visible(state, side, request):
    '''Reach is revealed in full - so the view is exactly the occupied part of it.'''
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces)
    assert seen(positions) == reachable(pieces) & occupied_squares(state.data)


# --------------------------------------------------------------------------- #
# Concrete views from the starting position
# --------------------------------------------------------------------------- #

def test_a_lone_rook_sees_its_rank_its_file_and_itself(state):
    positions = state.visible_positions({ROOK: ["a1"]})
    assert seen(positions) == {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",  # its rank
        "a2", "a7", "a8",                                 # the occupied squares of its file
    }


def test_a_lone_rook_reads_the_enemy_pieces_it_sees(state):
    positions = state.visible_positions({ROOK: ["a1"]})
    assert piece_at(positions, "a7") == PAWN
    assert piece_at(positions, "a8") == ROOK


def test_knights_see_only_the_pieces_on_their_landing_squares(state):
    '''b1 and g1 cover a3/c3/d2 and e2/f3/h3; only d2 and e2 are occupied.'''
    positions = state.visible_positions({KNIGHT: ["b1", "g1"]})
    assert seen(positions) == {"b1", "g1", "d2", "e2"}


def test_pawns_alone_see_nothing_of_the_enemy(state):
    '''Pawns on rank 2 only look at ranks 3 and 4, which are empty.'''
    positions = state.visible_positions({PAWN: state.white_pieces[PAWN]})
    assert seen(positions) == {f + "2" for f in "abcdefgh"}


def test_white_sees_only_the_files_its_long_range_pieces_stare_down(state, white):
    '''At the start those are the a, d and h files: the rooks and the queen.'''
    positions = state.visible_positions(white)
    enemy_camp = {coord for coord in seen(positions) if coord[1] in "78"}
    assert enemy_camp == {"a7", "a8", "d7", "d8", "h7", "h8"}


def test_most_of_the_enemy_camp_stays_hidden(state, white):
    positions = state.visible_positions(white)
    for coord in ["b7", "c7", "e7", "f7", "g7", "b8", "c8", "e8", "f8", "g8"]:
        assert piece_at(positions, coord) == EMPTY_SQUARE


# --------------------------------------------------------------------------- #
# Properties of this particular fog model
# --------------------------------------------------------------------------- #

def test_sight_passes_straight_through_blocking_pieces(state):
    '''The a1 rook reads a8 despite its own pawn standing on a2.

    Sight is computed from the move generators, which know nothing about
    occupancy, so nothing blocks a line.
    '''
    positions = state.visible_positions({ROOK: ["a1"]})
    assert piece_at(positions, "a2") == PAWN
    assert piece_at(positions, "a8") == ROOK


def test_a_view_does_not_say_which_side_a_piece_belongs_to(state):
    '''``data`` stores piece types only, so a seen enemy rook and your own rook
    are the same number.'''
    positions = state.visible_positions({ROOK: ["a1"]})
    assert piece_at(positions, "a8") == piece_at(positions, "a1") == ROOK


def test_an_empty_square_reads_the_same_whether_or_not_it_is_seen(state):
    '''a3 is watched by the a1 rook and c5 is watched by nobody; both read 0.

    Emptiness and ignorance share an encoding, so a view cannot express "I
    looked here and there was nothing".
    '''
    positions = state.visible_positions({ROOK: ["a1"]})
    assert piece_at(positions, "a3") == EMPTY_SQUARE
    assert piece_at(positions, "c5") == EMPTY_SQUARE


def test_neither_side_sees_the_whole_board_at_the_start(state, white, black):
    for pieces in (white, black):
        assert seen(state.visible_positions(pieces)) < occupied_squares(state.data)


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=NO_KING_IN_REGISTRY)
def test_a_full_side_can_be_projected(state):
    assert state.visible_positions(state.white_pieces).any()


@pytest.mark.xfail(strict=True, reason=NO_KING_IN_REGISTRY)
def test_gen_state_returns_a_view_for_each_side_and_the_truth(state):
    white_view, black_view, data = state.gen_state()
    assert white_view.shape == black_view.shape == (8, 8)
    assert data is state.data


@pytest.mark.xfail(strict=True, reason=NO_KING_IN_REGISTRY)
def test_a_king_reveals_the_squares_around_it(state):
    positions = state.visible_positions({KING: ["e1"]})
    assert piece_at(positions, "d1") == QUEEN
    assert piece_at(positions, "f1") == BISHOP
    assert piece_at(positions, "e2") == PAWN


@pytest.mark.xfail(
    strict=True,
    reason="pawn_moves only looks towards rank 8, so black pawns watch their own back rank",
)
def test_black_pawns_look_away_from_their_own_back_rank(state):
    '''Black's pawns should be watching rank 6, which is empty - so a pawn-only
    view of black should show rank 7 and nothing else. Today they look the wrong
    way and light up every piece on rank 8 instead.'''
    positions = state.visible_positions({PAWN: state.black_pieces[PAWN]})
    assert seen(positions) == {f + "7" for f in "abcdefgh"}
