'''The starting board state produced by ``ChessState.gen_game``.'''

import numpy as np
import pytest

from src.data.data import (
    BISHOP,
    COORDINATES,
    EMPTY_SQUARE,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    ChessState,
)

from .helpers import FILES, all_listed_squares, occupied_squares, piece_at, square

BACK_RANK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]


# --------------------------------------------------------------------------- #
# Piece codes
# --------------------------------------------------------------------------- #

def test_empty_square_is_zero():
    '''Emptiness is tested by equality with ``EMPTY_SQUARE``, and a zero default
    keeps a freshly zeroed array a legitimately empty board.'''
    assert EMPTY_SQUARE == 0


def test_piece_codes_are_distinct():
    codes = [EMPTY_SQUARE, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING]
    assert len(set(codes)) == len(codes)


def test_piece_codes_fit_in_a_small_integer_range():
    assert {PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING} == {1, 2, 3, 4, 5, 6}


# --------------------------------------------------------------------------- #
# The array representation
# --------------------------------------------------------------------------- #

def test_board_is_eight_by_eight_of_integers(board):
    assert board.shape == (8, 8)
    assert np.issubdtype(board.dtype, np.integer)


def test_board_holds_only_known_piece_codes(board):
    known = {EMPTY_SQUARE, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING}
    assert set(np.unique(board)).issubset(known)


def test_thirty_two_pieces_and_thirty_two_empty_squares(board):
    assert int(np.count_nonzero(board)) == 32
    assert int(np.count_nonzero(board == EMPTY_SQUARE)) == 32


@pytest.mark.parametrize("rank", ["3", "4", "5", "6"])
def test_middle_ranks_start_empty(board, rank):
    for file_ in FILES:
        assert piece_at(board, file_ + rank) == EMPTY_SQUARE


@pytest.mark.parametrize("row", [0, 7])
def test_back_ranks_hold_the_standard_arrangement(board, row):
    assert list(board[row]) == BACK_RANK


@pytest.mark.parametrize("row", [1, 6])
def test_pawn_ranks_are_all_pawns(board, row):
    assert list(board[row]) == [PAWN] * 8


@pytest.mark.parametrize(
    ("piece", "count"),
    [(PAWN, 16), (ROOK, 4), (KNIGHT, 4), (BISHOP, 4), (QUEEN, 2), (KING, 2)],
)
def test_piece_counts(board, piece, count):
    assert int(np.count_nonzero(board == piece)) == count


@pytest.mark.parametrize("rank", ["1", "8"])
def test_royalty_stands_on_the_d_and_e_files(board, rank):
    assert piece_at(board, "d" + rank) == QUEEN
    assert piece_at(board, "e" + rank) == KING


@pytest.mark.parametrize("corner", ["a1", "h1", "a8", "h8"])
def test_rooks_stand_in_the_corners(board, corner):
    assert piece_at(board, corner) == ROOK


@pytest.mark.parametrize("coord", ["b1", "g1", "b8", "g8"])
def test_knights_stand_beside_the_rooks(board, coord):
    assert piece_at(board, coord) == KNIGHT


@pytest.mark.parametrize("coord", ["c1", "f1", "c8", "f8"])
def test_bishops_stand_beside_the_knights(board, coord):
    assert piece_at(board, coord) == BISHOP


def test_board_array_carries_no_colour(board):
    '''The array stores piece *type* only, so the two halves are mirror images.

    Colour lives solely in ``white_pieces`` / ``black_pieces``; anything reading
    ``data`` alone cannot tell whose piece it is looking at.
    '''
    assert np.array_equal(board, board[::-1])


# --------------------------------------------------------------------------- #
# The dictionary representation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("colour", ["white_pieces", "black_pieces"])
def test_each_colour_lists_sixteen_pieces(state, colour):
    assert len(all_listed_squares(getattr(state, colour))) == 16


@pytest.mark.parametrize("colour", ["white_pieces", "black_pieces"])
def test_each_colour_lists_every_piece_type(state, colour):
    assert set(getattr(state, colour)) == {PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING}


@pytest.mark.parametrize(
    ("colour", "expected"),
    [
        (
            "white_pieces",
            {
                PAWN: [f + "2" for f in FILES],
                ROOK: ["a1", "h1"],
                KNIGHT: ["b1", "g1"],
                BISHOP: ["c1", "f1"],
                QUEEN: ["d1"],
                KING: ["e1"],
            },
        ),
        (
            "black_pieces",
            {
                PAWN: [f + "7" for f in FILES],
                ROOK: ["a8", "h8"],
                KNIGHT: ["b8", "g8"],
                BISHOP: ["c8", "f8"],
                QUEEN: ["d8"],
                KING: ["e8"],
            },
        ),
    ],
)
def test_starting_squares_per_colour(state, colour, expected):
    assert getattr(state, colour) == expected


@pytest.mark.parametrize("colour", ["white_pieces", "black_pieces"])
def test_listed_squares_are_real_squares(state, colour):
    for coord in all_listed_squares(getattr(state, colour)):
        assert coord in COORDINATES


@pytest.mark.parametrize("colour", ["white_pieces", "black_pieces"])
def test_no_colour_lists_a_square_twice(state, colour):
    listed = all_listed_squares(getattr(state, colour))
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize(
    ("colour", "ranks"), [("white_pieces", {"1", "2"}), ("black_pieces", {"7", "8"})]
)
def test_colours_start_on_their_own_side(state, colour, ranks):
    for coord in all_listed_squares(getattr(state, colour)):
        assert coord[1] in ranks


# --------------------------------------------------------------------------- #
# The two representations have to agree
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("colour", ["white_pieces", "black_pieces"])
def test_listed_pieces_stand_where_the_array_says(state, colour):
    for piece, squares in getattr(state, colour).items():
        for coord in squares:
            assert piece_at(state.data, coord) == piece


def test_the_two_colours_never_claim_the_same_square(state):
    white = set(all_listed_squares(state.white_pieces))
    black = set(all_listed_squares(state.black_pieces))
    assert white.isdisjoint(black)


def test_listed_squares_account_for_every_occupied_square(state):
    listed = set(all_listed_squares(state.white_pieces)) | set(
        all_listed_squares(state.black_pieces)
    )
    assert listed == occupied_squares(state.data)


def test_no_piece_is_listed_on_an_empty_square(state):
    for colour in (state.white_pieces, state.black_pieces):
        for coord in all_listed_squares(colour):
            assert piece_at(state.data, coord) != EMPTY_SQUARE


# --------------------------------------------------------------------------- #
# States are independent of one another
# --------------------------------------------------------------------------- #

def test_two_states_start_identical():
    first, second = ChessState(), ChessState()
    assert np.array_equal(first.data, second.data)
    assert first.white_pieces == second.white_pieces
    assert first.black_pieces == second.black_pieces


def test_mutating_one_board_does_not_touch_another():
    first, second = ChessState(), ChessState()
    first.data[COORDINATES["e2"]] = EMPTY_SQUARE
    assert piece_at(second.data, "e2") == PAWN


def test_mutating_one_piece_dictionary_does_not_touch_another():
    '''``gen_game`` builds fresh lists per call, so no two states share them.'''
    first, second = ChessState(), ChessState()
    first.white_pieces[PAWN].remove("e2")
    first.black_pieces[ROOK].append("e5")
    assert "e2" in second.white_pieces[PAWN]
    assert second.black_pieces[ROOK] == ["a8", "h8"]


def test_white_and_black_dictionaries_are_separate_objects(state):
    '''A capture mutates the opponent's lists in place; aliasing would corrupt
    the capturing side's own pieces.'''
    for piece in state.white_pieces:
        assert state.white_pieces[piece] is not state.black_pieces[piece]


# --------------------------------------------------------------------------- #
# Round-tripping the board back out
# --------------------------------------------------------------------------- #

def test_every_array_cell_maps_back_to_a_named_square(board):
    named = {square(row, col) for row in range(8) for col in range(8)}
    assert named == set(COORDINATES)
