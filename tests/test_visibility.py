'''The fog-of-war board state: what one side can see of ``ChessState.data``.

``visible_positions`` projects the full board down to the squares a side's own
pieces stand on, plus every square those pieces could move to. Anything else
comes back as ``EMPTY_SQUARE``.

``gen_state`` runs that for both sides.

Sight is computed from the move generators, which read the board, so a line of
sight stops at the first piece standing on it, and a pawn cannot see what blocks
its push.
'''

import numpy as np
import pytest

from src.data.data import (
    BISHOP,
    BLACK,
    COORDINATES,
    EMPTY_SQUARE,
    KING,
    KNIGHT,
    MOVE_FUNCTIONS,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
)

from .helpers import occupied_squares, piece_at, square

#: The colour each side fixture belongs to, which the generators move by.
COLOURS = {"white": WHITE, "black": BLACK}

OWN_PIECES_BLOCK_SIGHT = (
    "a piece cannot move onto one of its own, so a lone king watches none of "
    "the pieces standing next to it"
)


def without_king(pieces):
    '''A side's pieces minus the king, which the views below were written without.'''
    return {piece: list(squares) for piece, squares in pieces.items() if piece != KING}


def seen(positions):
    '''The squares a view reports something on.'''
    return {
        square(row, col)
        for row in range(8)
        for col in range(8)
        if positions[row, col] != EMPTY_SQUARE
    }


def reachable(state, pieces, colour):
    '''Own squares plus every square those pieces could move to.

    The generators take coordinates and read the board they are given, so this
    goes through ``COORDINATES`` on the way in and names the answers on the way
    back out, to compare against the squares a view reports.
    '''
    board = state.occupancy()

    squares = set()
    for piece, locations in pieces.items():
        for location in locations:
            squares.add(location)
            for move in MOVE_FUNCTIONS[piece](COORDINATES[location], board, colour):
                squares.add(square(*move))
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
    positions = state.visible_positions(white, WHITE)
    assert positions.shape == (8, 8)
    assert np.issubdtype(positions.dtype, np.integer)


def test_a_side_with_no_pieces_sees_nothing(state):
    assert not state.visible_positions({}, WHITE).any()


def test_a_view_is_a_fresh_array_each_time(state, white):
    first = state.visible_positions(white, WHITE)
    second = state.visible_positions(white, WHITE)
    assert first is not second
    assert np.array_equal(first, second)


def test_building_a_view_does_not_disturb_the_board(state, white):
    before = state.data.copy()
    state.visible_positions(white, WHITE)
    assert np.array_equal(state.data, before)


def test_building_a_view_does_not_disturb_the_pieces(state):
    before = {piece: list(squares) for piece, squares in state.white_pieces.items()}
    state.visible_positions(without_king(state.white_pieces), WHITE)
    assert state.white_pieces == before


# --------------------------------------------------------------------------- #
# What a view is allowed to contain
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("side", ["white", "black"])
def test_a_view_never_invents_a_piece(state, side, request):
    '''Every square the view reports holds exactly what the real board holds.'''
    positions = state.visible_positions(request.getfixturevalue(side), COLOURS[side])
    for coord in seen(positions):
        assert piece_at(positions, coord) == piece_at(state.data, coord)


@pytest.mark.parametrize("side", ["white", "black"])
def test_your_own_pieces_are_always_visible(state, side, request):
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces, COLOURS[side])
    for piece, locations in pieces.items():
        for location in locations:
            assert piece_at(positions, location) == piece


@pytest.mark.parametrize("side", ["white", "black"])
def test_nothing_outside_reach_is_visible(state, side, request):
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces, COLOURS[side])
    assert seen(positions) <= reachable(state, pieces, COLOURS[side])


@pytest.mark.parametrize("side", ["white", "black"])
def test_everything_within_reach_is_visible(state, side, request):
    '''Reach is revealed in full - so the view is exactly the occupied part of it.'''
    pieces = request.getfixturevalue(side)
    positions = state.visible_positions(pieces, COLOURS[side])
    assert seen(positions) == reachable(state, pieces, COLOURS[side]) & occupied_squares(state.data)


# --------------------------------------------------------------------------- #
# Concrete views from the starting position
# --------------------------------------------------------------------------- #

def test_pawns_alone_see_nothing_of_the_enemy(state):
    '''Pawns on rank 2 only look at ranks 3 and 4, which are empty.'''
    positions = state.visible_positions({PAWN: state.white_pieces[PAWN]}, WHITE)
    assert seen(positions) == {f + "2" for f in "abcdefgh"}


def test_most_of_the_enemy_camp_stays_hidden(state, white):
    positions = state.visible_positions(white, WHITE)
    for coord in ["b7", "c7", "e7", "f7", "g7", "b8", "c8", "e8", "f8", "g8"]:
        assert piece_at(positions, coord) == EMPTY_SQUARE


# --------------------------------------------------------------------------- #
# A pawn is blind to what blocks it
#
# "An opposing piece blocking a pawn's forward move is hidden unless another
# piece reveals it" - which falls out of sight being the squares you may move
# to, since a blocked push is not one of them.
# --------------------------------------------------------------------------- #

def lone(state, white, black):
    '''Replace the board with a handful of pieces, and return white's view.'''
    state.white_pieces, state.black_pieces = white, black
    state.data = state.write_board()
    return state.visible_positions(white, WHITE)


def test_a_pawn_does_not_see_the_pawn_blocking_it(state):
    positions = lone(state, {PAWN: ["e4"], KING: ["e1"]}, {PAWN: ["e5"], KING: ["e8"]})
    assert piece_at(positions, "e5") == EMPTY_SQUARE
    assert seen(positions) == {"e1", "e4"}


def test_a_pawn_does_not_see_what_blocks_its_double_push(state):
    positions = lone(state, {PAWN: ["e2"], KING: ["e1"]}, {PAWN: ["e4"], KING: ["e8"]})
    assert piece_at(positions, "e4") == EMPTY_SQUARE


def test_another_piece_reveals_the_blocking_pawn(state):
    '''A knight on d3 attacks e5, so the pawn the e4 pawn cannot see is read.'''
    positions = lone(
        state, {PAWN: ["e4"], KNIGHT: ["d3"], KING: ["e1"]}, {PAWN: ["e5"], KING: ["e8"]}
    )
    assert piece_at(positions, "e5") == PAWN


def test_a_piece_stuck_behind_the_pawn_reveals_nothing(state):
    '''The rook's own pawn ends its line on e4, short of the pawn on e5.'''
    positions = lone(
        state, {PAWN: ["e4"], ROOK: ["e3"], KING: ["a1"]}, {PAWN: ["e5"], KING: ["e8"]}
    )
    assert piece_at(positions, "e5") == EMPTY_SQUARE


def test_a_pawn_still_sees_what_it_can_take(state):
    '''Blocked ahead, but the diagonals are captures and so are visible.'''
    positions = lone(
        state, {PAWN: ["e4"], KING: ["e1"]}, {PAWN: ["e5", "d5", "f5"], KING: ["e8"]}
    )
    assert piece_at(positions, "e5") == EMPTY_SQUARE
    assert piece_at(positions, "d5") == PAWN
    assert piece_at(positions, "f5") == PAWN


# --------------------------------------------------------------------------- #
# Properties of this particular fog model
# --------------------------------------------------------------------------- #

def test_an_empty_square_reads_the_same_whether_or_not_it_is_seen(state):
    '''a3 is watched by the a1 rook and c5 is watched by nobody; both read 0.

    Emptiness and ignorance share an encoding, so a view cannot express "I
    looked here and there was nothing".
    '''
    positions = state.visible_positions({ROOK: ["a1"]}, WHITE)
    assert piece_at(positions, "a3") == EMPTY_SQUARE
    assert piece_at(positions, "c5") == EMPTY_SQUARE


def test_neither_side_sees_the_whole_board_at_the_start(state, white, black):
    for pieces, colour in ((white, WHITE), (black, BLACK)):
        assert seen(state.visible_positions(pieces, colour)) < occupied_squares(state.data)


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

def test_a_full_side_can_be_projected(state):
    assert state.visible_positions(state.white_pieces, WHITE).any()


def test_gen_state_returns_a_view_for_each_side_and_the_truth(state):
    white_view, black_view, data = state.gen_state()
    assert white_view.shape == black_view.shape == (8, 8)
    assert data is state.data


@pytest.mark.xfail(strict=True, reason=OWN_PIECES_BLOCK_SIGHT)
def test_a_king_reveals_the_squares_around_it(state):
    positions = state.visible_positions({KING: ["e1"]}, WHITE)
    assert piece_at(positions, "d1") == QUEEN
    assert piece_at(positions, "f1") == BISHOP
    assert piece_at(positions, "e2") == PAWN


def test_black_pawns_look_away_from_their_own_back_rank(state):
    '''Black's pawns watch rank 6, which is empty - so a pawn-only view of
    black shows rank 7 and nothing else.'''
    positions = state.visible_positions({PAWN: state.black_pieces[PAWN]}, BLACK)
    assert seen(positions) == {f + "7" for f in "abcdefgh"}
