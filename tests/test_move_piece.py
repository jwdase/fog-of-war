'''State transitions through ``ChessState.move_piece``.

``move_piece`` is the only thing in ``data.py`` that advances a board state, and
it does not currently run: every call raises ``NameError`` on its first line,
where it tests ``piece not in pieces`` against a name that is never bound.

Every test below is therefore marked ``xfail(strict=True)`` and describes the
transition the method is meant to perform, with a reason naming the specific
defect in the way. They are a specification: fix a defect and the matching test
starts passing, which a strict xfail reports as a failure telling you to drop the
marker. Each ``reason`` is the bug it is waiting on.

Defects these tests pin down, in the order they bite:

1. ``pieces`` is undefined - the lookups want ``player_pieces``.
2. ``coordinates`` holds a ``(row, column)`` tuple from ``COORDINATES`` but is
   compared against the algebraic strings the generators return, so the
   ``in move_function(location)`` membership test can never be true.
3. The capture branch removes the opponent's piece and then raises
   ``ValueError`` unconditionally, so a capture can never succeed - and the
   raise also skips updating the board.
4. Nothing ever appends ``new_coordinates`` to ``player_pieces[piece]``, so a
   moved piece is dropped out of its own side's dictionary.
'''

import pytest

from src.data.data import EMPTY_SQUARE, KNIGHT, PAWN, QUEEN, ROOK

from .helpers import piece_at

UNDEFINED_PIECES = "move_piece reads an undefined name `pieces`"
TUPLE_VS_STRING = "move_piece compares a (row, column) tuple against algebraic strings"
CAPTURE_ALWAYS_RAISES = "the capture branch raises unconditionally"
NO_DESTINATION_APPEND = "move_piece never appends the destination to player_pieces"


def move(state, piece, new_coordinates, column=None, capture=False):
    '''Move a white piece, passing white as the player and black as the opponent.'''
    return state.move_piece(
        state.white_pieces,
        state.black_pieces,
        piece,
        column,
        new_coordinates,
        capture,
    )


# --------------------------------------------------------------------------- #
# A quiet move
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_pawn_push_empties_the_square_it_came_from(state):
    move(state, PAWN, "e4")
    assert piece_at(state.data, "e2") == EMPTY_SQUARE


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_pawn_push_fills_the_square_it_went_to(state):
    move(state, PAWN, "e4")
    assert piece_at(state.data, "e4") == PAWN


@pytest.mark.xfail(strict=True, reason=NO_DESTINATION_APPEND)
def test_a_pawn_push_is_recorded_in_the_movers_dictionary(state):
    move(state, PAWN, "e4")
    assert "e2" not in state.white_pieces[PAWN]
    assert "e4" in state.white_pieces[PAWN]


@pytest.mark.xfail(strict=True, reason=NO_DESTINATION_APPEND)
def test_a_quiet_move_changes_neither_sides_piece_count(state):
    move(state, PAWN, "e4")
    assert len(state.white_pieces[PAWN]) == 8
    assert len(state.black_pieces[PAWN]) == 8


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_quiet_move_leaves_thirty_two_pieces_on_the_board(state):
    move(state, PAWN, "e4")
    assert int((state.data != EMPTY_SQUARE).sum()) == 32


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_knight_is_resolved_from_the_only_square_that_reaches(state):
    '''Of the two white knights, only g1 can reach f3.'''
    move(state, KNIGHT, "f3")
    assert piece_at(state.data, "g1") == EMPTY_SQUARE
    assert piece_at(state.data, "b1") == KNIGHT
    assert piece_at(state.data, "f3") == KNIGHT


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_quiet_move_does_not_touch_the_opponent(state):
    before = dict(state.black_pieces)
    move(state, KNIGHT, "f3")
    assert state.black_pieces == before


# --------------------------------------------------------------------------- #
# A capture
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=CAPTURE_ALWAYS_RAISES)
def test_a_capture_removes_the_taken_piece_from_the_opponent(state):
    '''Rxa8: the a1 rook takes the black rook on a8.'''
    move(state, ROOK, "a8", capture=True)
    assert "a8" not in state.black_pieces[ROOK]
    assert state.black_pieces[ROOK] == ["h8"]


@pytest.mark.xfail(strict=True, reason=CAPTURE_ALWAYS_RAISES)
def test_a_capture_puts_the_capturing_piece_on_the_square(state):
    move(state, ROOK, "a8", capture=True)
    assert piece_at(state.data, "a1") == EMPTY_SQUARE
    assert piece_at(state.data, "a8") == ROOK
    assert "a8" in state.white_pieces[ROOK]


@pytest.mark.xfail(strict=True, reason=CAPTURE_ALWAYS_RAISES)
def test_a_capture_leaves_thirty_one_pieces_on_the_board(state):
    move(state, ROOK, "a8", capture=True)
    assert int((state.data != EMPTY_SQUARE).sum()) == 31


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_capturing_an_empty_square_is_refused(state):
    with pytest.raises(ValueError, match="capture"):
        move(state, PAWN, "e4", capture=True)


@pytest.mark.xfail(strict=True, reason=CAPTURE_ALWAYS_RAISES)
def test_a_refused_capture_leaves_the_board_untouched(state):
    with pytest.raises(ValueError):
        move(state, PAWN, "e4", capture=True)
    assert piece_at(state.data, "e2") == PAWN
    assert piece_at(state.data, "e4") == EMPTY_SQUARE
    assert len(state.white_pieces[PAWN]) == 8


# --------------------------------------------------------------------------- #
# Ambiguity and rejection
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_two_candidates_without_a_file_hint_are_refused(state):
    '''Both rooks "reach" d1 on an empty board, so Rd1 is ambiguous.'''
    with pytest.raises(ValueError, match="Multiple pieces"):
        move(state, ROOK, "d1")


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_a_file_hint_picks_between_two_candidates(state):
    '''Rad1: the file hint selects the a1 rook.'''
    move(state, ROOK, "d1", column="a")
    assert piece_at(state.data, "a1") == EMPTY_SQUARE
    assert piece_at(state.data, "h1") == ROOK


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_a_destination_no_piece_reaches_is_refused(state):
    with pytest.raises(ValueError, match="No piece of type"):
        move(state, KNIGHT, "h6")


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_a_file_hint_that_matches_nothing_is_refused(state):
    with pytest.raises(ValueError, match="No piece of type"):
        move(state, KNIGHT, "f3", column="b")


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_a_piece_the_player_does_not_own_is_refused(state):
    state.white_pieces.pop(QUEEN)
    with pytest.raises(ValueError, match="not found in pieces"):
        move(state, QUEEN, "d4")


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_an_empty_square_is_not_a_movable_piece(state):
    with pytest.raises(ValueError):
        move(state, EMPTY_SQUARE, "e4")


@pytest.mark.xfail(strict=True, reason=UNDEFINED_PIECES)
def test_a_rejected_move_leaves_the_board_untouched(state):
    before = state.data.copy()
    with pytest.raises(ValueError):
        move(state, KNIGHT, "h6")
    assert (state.data == before).all()
    assert state.white_pieces[KNIGHT] == ["b1", "g1"]


# --------------------------------------------------------------------------- #
# The per-colour wrappers
#
# Note the argument order differs from ``move_piece`` itself:
# ``move_piece_white(piece, new_coordinates, capture, column)``.
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_the_white_wrapper_moves_a_white_piece(state):
    state.move_piece_white(PAWN, "e4", False, None)
    assert piece_at(state.data, "e2") == EMPTY_SQUARE
    assert piece_at(state.data, "e4") == PAWN


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_the_black_wrapper_moves_a_black_piece(state):
    '''Nf6: of the two black knights, only g8 can reach f6.'''
    state.move_piece_black(KNIGHT, "f6", False, None)
    assert piece_at(state.data, "g8") == EMPTY_SQUARE
    assert piece_at(state.data, "f6") == KNIGHT


@pytest.mark.xfail(strict=True, reason=TUPLE_VS_STRING)
def test_the_black_wrapper_leaves_white_alone(state):
    '''The wrappers differ only in which dictionary is the player's and which
    the opponent's, so a swap here would move the wrong side's pieces.'''
    before = {piece: list(squares) for piece, squares in state.white_pieces.items()}
    state.move_piece_black(KNIGHT, "f6", False, None)
    assert state.white_pieces == before
    assert "f6" in state.black_pieces[KNIGHT]


@pytest.mark.xfail(
    strict=True,
    reason="pawn_moves only looks towards rank 8, so no black pawn resolves to e5",
)
def test_the_black_wrapper_can_push_a_pawn(state):
    state.move_piece_black(PAWN, "e5", False, None)
    assert piece_at(state.data, "e7") == EMPTY_SQUARE
    assert piece_at(state.data, "e5") == PAWN
