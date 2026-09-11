'''The registries that turn algebraic notation into board-state operations.'''

import pytest

from src.data.data import (
    BISHOP,
    BLACK,
    EMPTY_SQUARE,
    KING,
    KNIGHT,
    MOVER,
    PAWN,
    PIECES,
    QUEEN,
    ROOK,
    WHITE,
    ChessState,
)


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #

def test_the_two_players_are_distinct():
    assert WHITE != BLACK


def test_every_player_has_a_mover():
    assert set(MOVER) == {WHITE, BLACK}


@pytest.mark.parametrize(
    ("player", "method"),
    [(WHITE, "move_piece_white"), (BLACK, "move_piece_black")],
)
def test_the_mover_dispatches_to_the_right_side(player, method):
    assert MOVER[player] is getattr(ChessState, method)


# --------------------------------------------------------------------------- #
# Piece letters
# --------------------------------------------------------------------------- #

def test_letters_cover_the_pieces_that_are_written_out():
    assert PIECES == {"N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING}


def test_every_letter_is_uppercase():
    '''``ChessGame.make_move`` reads a leading lowercase character as a pawn move,
    so a lowercase key here would be unreachable.'''
    assert all(letter.isupper() for letter in PIECES)


def test_a_pawn_has_no_letter():
    '''Pawn moves are written as a bare square ("e4"), never "Pe4".'''
    assert PAWN not in PIECES.values()


def test_letters_map_to_distinct_pieces():
    assert len(set(PIECES.values())) == len(PIECES)


def test_no_letter_maps_to_an_empty_square():
    assert EMPTY_SQUARE not in PIECES.values()
