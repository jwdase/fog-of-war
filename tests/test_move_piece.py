'''State transitions through ``ChessState.move_piece``.

``move_piece`` is the only thing in ``data.py`` that advances a board state. It
is handed what algebraic notation records - a piece type, a destination, and at
most a file hint - and works out which piece it was that moved, by asking the
move generators which of that side's pieces can reach the destination on the
board as it stands. Two candidates are an error, and so is none.

Notation is the boundary: the piece dictionaries and ``new_location`` are
squares ("e4"), while the generators are asked in coordinates.

Because the generators read the board, a move through a piece no longer
resolves - which is why the captures below are played into a real position
rather than conjured out of the starting one.

One move it still cannot resolve, marked xfail at the end: castling, which
``ChessGame.make_move`` hands it as a two-square king move no generator offers.

En passant needs no history here. The games are records of legal chess, so a
pawn that captures onto an empty square can only be capturing en passant, and
the pawn it takes can only be the one alongside it.
'''

import pytest

from src.data.data import BISHOP, EMPTY_SQUARE, KING, KNIGHT, PAWN, QUEEN, ROOK

from .helpers import piece_at


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


@pytest.fixture
def spanish(state):
    '''1.e4 e5 2.Nf3 Nc6 3.Bb5, where the bishop can take the knight on c6.

    Nothing is en prise in the starting position, so a capture has to be played
    into.
    '''
    move(state, PAWN, "e4")
    state.move_piece_black(PAWN, "e5", False, None)
    move(state, KNIGHT, "f3")
    state.move_piece_black(KNIGHT, "c6", False, None)
    move(state, BISHOP, "b5")
    return state


@pytest.fixture
def two_rooks(state):
    '''Both white rooks on an empty fourth rank, where either can reach d4.'''
    state.white_pieces[ROOK] = ["a4", "h4"]
    state.data = state.write_board()
    return state


# --------------------------------------------------------------------------- #
# A quiet move
# --------------------------------------------------------------------------- #

def test_a_pawn_push_empties_the_square_it_came_from(state):
    move(state, PAWN, "e4")
    assert piece_at(state.data, "e2") == EMPTY_SQUARE


def test_a_pawn_push_fills_the_square_it_went_to(state):
    move(state, PAWN, "e4")
    assert piece_at(state.data, "e4") == PAWN


def test_a_pawn_push_is_recorded_in_the_movers_dictionary(state):
    move(state, PAWN, "e4")
    assert "e2" not in state.white_pieces[PAWN]
    assert "e4" in state.white_pieces[PAWN]


def test_a_quiet_move_changes_neither_sides_piece_count(state):
    move(state, PAWN, "e4")
    assert len(state.white_pieces[PAWN]) == 8
    assert len(state.black_pieces[PAWN]) == 8


def test_a_quiet_move_leaves_thirty_two_pieces_on_the_board(state):
    move(state, PAWN, "e4")
    assert int((state.data != EMPTY_SQUARE).sum()) == 32


def test_a_knight_is_resolved_from_the_only_square_that_reaches(state):
    '''Of the two white knights, only g1 can reach f3.'''
    move(state, KNIGHT, "f3")
    assert piece_at(state.data, "g1") == EMPTY_SQUARE
    assert piece_at(state.data, "b1") == KNIGHT
    assert piece_at(state.data, "f3") == KNIGHT


def test_a_quiet_move_does_not_touch_the_opponent(state):
    before = dict(state.black_pieces)
    move(state, KNIGHT, "f3")
    assert state.black_pieces == before


def test_a_piece_cannot_move_through_another(state):
    '''The board is consulted, so the a1 rook is shut in by its own pawn.'''
    with pytest.raises(ValueError, match="No piece of type"):
        move(state, ROOK, "a4")


# --------------------------------------------------------------------------- #
# A capture
# --------------------------------------------------------------------------- #

def test_a_capture_removes_the_taken_piece_from_the_opponent(spanish):
    '''Bxc6: the b5 bishop takes the black knight on c6.'''
    move(spanish, BISHOP, "c6", capture=True)
    assert "c6" not in spanish.black_pieces[KNIGHT]
    assert spanish.black_pieces[KNIGHT] == ["g8"]


def test_a_capture_puts_the_capturing_piece_on_the_square(spanish):
    move(spanish, BISHOP, "c6", capture=True)
    assert piece_at(spanish.data, "b5") == EMPTY_SQUARE
    assert piece_at(spanish.data, "c6") == BISHOP
    assert "c6" in spanish.white_pieces[BISHOP]


def test_a_capture_leaves_thirty_one_pieces_on_the_board(spanish):
    move(spanish, BISHOP, "c6", capture=True)
    assert int((spanish.data != EMPTY_SQUARE).sum()) == 31


def test_a_capture_is_resolved_even_though_the_square_is_occupied(spanish):
    '''An enemy piece ends a line rather than blocking it, so c6 is reachable.'''
    move(spanish, BISHOP, "c6", capture=True)
    assert spanish.white_pieces[BISHOP] == ["c1", "c6"]


def test_capturing_an_empty_square_is_refused(state):
    with pytest.raises(ValueError, match="capture"):
        move(state, PAWN, "e4", capture=True)


def test_a_refused_capture_leaves_the_board_untouched(state):
    with pytest.raises(ValueError):
        move(state, PAWN, "e4", capture=True)
    assert piece_at(state.data, "e2") == PAWN
    assert piece_at(state.data, "e4") == EMPTY_SQUARE
    assert len(state.white_pieces[PAWN]) == 8


# --------------------------------------------------------------------------- #
# Ambiguity and rejection
# --------------------------------------------------------------------------- #

def test_two_candidates_without_a_file_hint_are_refused(two_rooks):
    '''Both rooks reach d4 down an empty fourth rank, so Rd4 is ambiguous.'''
    with pytest.raises(ValueError, match="Multiple pieces"):
        move(two_rooks, ROOK, "d4")


def test_a_file_hint_picks_between_two_candidates(two_rooks):
    '''Rad4: the file hint selects the a4 rook.'''
    move(two_rooks, ROOK, "d4", column="a")
    assert piece_at(two_rooks.data, "a4") == EMPTY_SQUARE
    assert piece_at(two_rooks.data, "d4") == ROOK
    assert piece_at(two_rooks.data, "h4") == ROOK


def test_a_destination_no_piece_reaches_is_refused(state):
    with pytest.raises(ValueError, match="No piece of type"):
        move(state, KNIGHT, "h6")


def test_a_file_hint_that_matches_nothing_is_refused(state):
    with pytest.raises(ValueError, match="No piece of type"):
        move(state, KNIGHT, "f3", column="b")


def test_a_piece_the_player_does_not_own_is_refused(state):
    state.white_pieces.pop(QUEEN)
    with pytest.raises(ValueError, match="not found in pieces"):
        move(state, QUEEN, "d4")


def test_an_empty_square_is_not_a_movable_piece(state):
    with pytest.raises(ValueError):
        move(state, EMPTY_SQUARE, "e4")


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

def test_the_white_wrapper_moves_a_white_piece(state):
    state.move_piece_white(PAWN, "e4", False, None)
    assert piece_at(state.data, "e2") == EMPTY_SQUARE
    assert piece_at(state.data, "e4") == PAWN


def test_the_black_wrapper_moves_a_black_piece(state):
    '''Nf6: of the two black knights, only g8 can reach f6.'''
    state.move_piece_black(KNIGHT, "f6", False, None)
    assert piece_at(state.data, "g8") == EMPTY_SQUARE
    assert piece_at(state.data, "f6") == KNIGHT


def test_the_black_wrapper_leaves_white_alone(state):
    '''The wrappers differ only in which dictionary is the player's and which
    the opponent's, so a swap here would move the wrong side's pieces.'''
    before = {piece: list(squares) for piece, squares in state.white_pieces.items()}
    state.move_piece_black(KNIGHT, "f6", False, None)
    assert state.white_pieces == before
    assert "f6" in state.black_pieces[KNIGHT]


def test_the_black_wrapper_can_push_a_pawn(state):
    '''The wrapper names black, so the pawn is pushed towards rank 1.'''
    state.move_piece_black(PAWN, "e5", False, None)
    assert piece_at(state.data, "e7") == EMPTY_SQUARE
    assert piece_at(state.data, "e5") == PAWN


def test_the_wrappers_push_their_own_pawns_in_opposite_directions(state):
    state.move_piece_white(PAWN, "d4", False, None)
    state.move_piece_black(PAWN, "d5", False, None)
    assert "d4" in state.white_pieces[PAWN]
    assert "d5" in state.black_pieces[PAWN]


# --------------------------------------------------------------------------- #
# Known gaps - moves ``move_piece`` still cannot resolve
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="castling is handed to move_piece as a two-square king move, which "
    "king_moves does not offer",
)
def test_castling_can_be_resolved(state):
    state.white_pieces[KNIGHT].remove("g1")
    state.white_pieces[BISHOP].remove("f1")
    state.data = state.write_board()
    state.move_piece_white(KING, "g1", False, None)
    assert piece_at(state.data, "g1") == KING


# --------------------------------------------------------------------------- #
# En passant
#
# The one capture that lands on an empty square and empties another.
# --------------------------------------------------------------------------- #

@pytest.fixture
def passed(state):
    '''White pawn on e5, black pawn alongside on d5 as if just pushed there.'''
    state.white_pieces[PAWN] = ["e5"]
    state.black_pieces[PAWN] = ["d5"]
    state.data = state.write_board()
    return state


def test_en_passant_moves_the_capturing_pawn_to_the_empty_square(passed):
    passed.move_piece_white(PAWN, "d6", True, "e")
    assert passed.white_pieces[PAWN] == ["d6"]
    assert piece_at(passed.data, "d6") == PAWN
    assert piece_at(passed.data, "e5") == EMPTY_SQUARE


def test_en_passant_takes_the_pawn_alongside_not_the_destination(passed):
    passed.move_piece_white(PAWN, "d6", True, "e")
    assert passed.black_pieces[PAWN] == []
    assert piece_at(passed.data, "d5") == EMPTY_SQUARE


def test_en_passant_removes_exactly_one_piece(passed):
    before = int((passed.data != EMPTY_SQUARE).sum())
    passed.move_piece_white(PAWN, "d6", True, "e")
    assert int((passed.data != EMPTY_SQUARE).sum()) == before - 1


def test_black_can_take_en_passant_too(state):
    '''The taken pawn is alongside in the other direction.'''
    state.white_pieces[PAWN] = ["e4"]
    state.black_pieces[PAWN] = ["d4"]
    state.data = state.write_board()
    state.move_piece_black(PAWN, "e3", True, "d")
    assert state.black_pieces[PAWN] == ["e3"]
    assert state.white_pieces[PAWN] == []
    assert piece_at(state.data, "e4") == EMPTY_SQUARE


def test_an_ordinary_pawn_capture_still_takes_what_it_lands_on(state):
    state.white_pieces[PAWN] = ["e4"]
    state.black_pieces[PAWN] = ["d5"]
    state.data = state.write_board()
    state.move_piece_white(PAWN, "d5", True, "e")
    assert state.white_pieces[PAWN] == ["d5"]
    assert state.black_pieces[PAWN] == []


def test_a_pawn_capture_onto_an_empty_square_with_nothing_alongside_is_refused(state):
    '''Read as en passant, and there is no pawn to take.'''
    with pytest.raises(ValueError, match="capture"):
        move(state, PAWN, "e4", capture=True)
