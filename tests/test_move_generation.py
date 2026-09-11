'''The move generators that board states are advanced and revealed with.

These are pure geometry: they answer "which squares could this piece reach on an
otherwise empty board", take no colour and no board, and so never consult the
occupancy of ``ChessState.data``.
'''

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
    bishop_moves,
    king_moves,
    knight_moves,
    pawn_moves,
    queen_moves,
    rook_moves,
)

from .helpers import ALL_SQUARES, FILES

GENERATORS = [
    ("pawn", pawn_moves),
    ("knight", knight_moves),
    ("bishop", bishop_moves),
    ("rook", rook_moves),
    ("queen", queen_moves),
    ("king", king_moves),
]
GENERATOR_IDS = [name for name, _ in GENERATORS]
SLIDERS = [("bishop", bishop_moves), ("rook", rook_moves), ("queen", queen_moves)]
STEPPERS = [("knight", knight_moves), ("king", king_moves)]
CENTRE = ["d4", "d5", "e4", "e5"]
CORNERS = ["a1", "h1", "a8", "h8"]


def offsets(origin, destination):
    '''Row/column delta between two algebraic squares.'''
    x0, y0 = COORDINATES[origin]
    x1, y1 = COORDINATES[destination]
    return x1 - x0, y1 - y0


# --------------------------------------------------------------------------- #
# Invariants every generator has to satisfy, from every square
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_moves_stay_on_the_board(_name, generator, coord):
    for destination in generator(coord):
        assert destination in COORDINATES


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_moves_are_not_repeated(_name, generator, coord):
    moves = generator(coord)
    assert len(moves) == len(set(moves))


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_piece_never_moves_to_its_own_square(_name, generator, coord):
    assert coord not in generator(coord)


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_generators_are_pure(_name, generator, coord):
    '''Calling a generator twice gives the same answer and mutates no state.'''
    assert generator(coord) == generator(coord)


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
def test_an_unknown_square_is_rejected(_name, generator):
    with pytest.raises(KeyError):
        generator("j9")


# --------------------------------------------------------------------------- #
# Knight
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_knight_moves_are_all_l_shaped(coord):
    for destination in knight_moves(coord):
        assert sorted(abs(d) for d in offsets(coord, destination)) == [1, 2]


@pytest.mark.parametrize(
    ("coord", "expected"),
    [
        ("d4", {"c2", "e2", "b3", "f3", "b5", "f5", "c6", "e6"}),
        ("b1", {"a3", "c3", "d2"}),
        ("g8", {"e7", "f6", "h6"}),
        ("a1", {"b3", "c2"}),
        ("h8", {"f7", "g6"}),
    ],
)
def test_knight_move_sets(coord, expected):
    assert set(knight_moves(coord)) == expected


@pytest.mark.parametrize("coord", CENTRE)
def test_a_central_knight_has_eight_moves(coord):
    assert len(knight_moves(coord)) == 8


@pytest.mark.parametrize("coord", CORNERS)
def test_a_cornered_knight_has_two_moves(coord):
    assert len(knight_moves(coord)) == 2


# --------------------------------------------------------------------------- #
# King
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_king_only_steps_one_square(coord):
    for destination in king_moves(coord):
        dx, dy = offsets(coord, destination)
        assert max(abs(dx), abs(dy)) == 1


@pytest.mark.parametrize(
    ("coord", "expected"),
    [
        ("e1", {"d1", "f1", "d2", "e2", "f2"}),
        ("a8", {"a7", "b7", "b8"}),
        ("d4", {"c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"}),
    ],
)
def test_king_move_sets(coord, expected):
    assert set(king_moves(coord)) == expected


@pytest.mark.parametrize("coord", CENTRE)
def test_a_central_king_has_eight_moves(coord):
    assert len(king_moves(coord)) == 8


@pytest.mark.parametrize("coord", CORNERS)
def test_a_cornered_king_has_three_moves(coord):
    assert len(king_moves(coord)) == 3


@pytest.mark.parametrize("coord", ["e1", "e8", "a4", "h4"])
def test_a_king_on_an_edge_has_five_moves(coord):
    assert len(king_moves(coord)) == 5


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_king_reaches_every_neighbouring_square(coord):
    row, col = COORDINATES[coord]
    neighbours = {
        (row + dx, col + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    }
    on_board = {n for n in neighbours if 0 <= n[0] < 8 and 0 <= n[1] < 8}
    assert {COORDINATES[m] for m in king_moves(coord)} == on_board


# --------------------------------------------------------------------------- #
# Bishop
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_bishop_moves_stay_on_a_diagonal(coord):
    for destination in bishop_moves(coord):
        dx, dy = offsets(coord, destination)
        assert abs(dx) == abs(dy) != 0


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_bishop_keeps_its_square_colour(coord):
    '''Diagonal movement preserves the parity of row + column.'''
    row, col = COORDINATES[coord]
    for destination in bishop_moves(coord):
        new_row, new_col = COORDINATES[destination]
        assert (new_row + new_col) % 2 == (row + col) % 2


@pytest.mark.parametrize(
    ("coord", "expected"),
    [
        ("a1", {"b2", "c3", "d4", "e5", "f6", "g7", "h8"}),
        ("h1", {"g2", "f3", "e4", "d5", "c6", "b7", "a8"}),
        ("c1", {"b2", "a3", "d2", "e3", "f4", "g5", "h6"}),
    ],
)
def test_bishop_move_sets(coord, expected):
    assert set(bishop_moves(coord)) == expected


@pytest.mark.parametrize("coord", CENTRE)
def test_a_central_bishop_has_thirteen_moves(coord):
    assert len(bishop_moves(coord)) == 13


@pytest.mark.parametrize("coord", CORNERS)
def test_a_cornered_bishop_sweeps_the_long_diagonal(coord):
    assert len(bishop_moves(coord)) == 7


# --------------------------------------------------------------------------- #
# Rook
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_rook_moves_share_a_rank_or_a_file(coord):
    for destination in rook_moves(coord):
        dx, dy = offsets(coord, destination)
        assert (dx == 0) != (dy == 0)


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_rook_always_has_fourteen_moves(coord):
    assert len(rook_moves(coord)) == 14


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_rook_reaches_its_whole_rank_and_file(coord):
    file_, rank = coord[0], coord[1]
    whole_cross = {f + rank for f in FILES} | {file_ + r for r in "12345678"}
    assert set(rook_moves(coord)) == whole_cross - {coord}


# --------------------------------------------------------------------------- #
# Queen
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_queen_combines_bishop_and_rook(coord):
    assert set(queen_moves(coord)) == set(bishop_moves(coord)) | set(rook_moves(coord))


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_bishop_and_rook_reach_disjoint_squares(coord):
    '''Which is why concatenating them cannot introduce a duplicate.'''
    assert set(bishop_moves(coord)).isdisjoint(set(rook_moves(coord)))
    assert len(queen_moves(coord)) == len(bishop_moves(coord)) + len(rook_moves(coord))


@pytest.mark.parametrize("coord", CENTRE)
def test_a_central_queen_has_twenty_seven_moves(coord):
    assert len(queen_moves(coord)) == 27


@pytest.mark.parametrize("coord", CORNERS)
def test_a_cornered_queen_has_twenty_one_moves(coord):
    assert len(queen_moves(coord)) == 21


# --------------------------------------------------------------------------- #
# Reciprocity, and blindness to occupancy
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("_name", "generator"), SLIDERS + STEPPERS, ids=["bishop", "rook", "queen", "knight", "king"]
)
@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_moves_are_reciprocal(_name, generator, coord):
    '''Every generator except the pawn is symmetric: if a can reach b, b can reach a.'''
    for destination in generator(coord):
        assert coord in generator(destination)


def test_sliding_moves_ignore_blocking_pieces():
    '''A rook on a1 "reaches" a8 even though six pieces stand in the way.

    The generators describe an empty board, so whatever resolves a move - or
    computes what a player can see - has to apply occupancy itself.
    '''
    assert "a8" in rook_moves("a1")
    assert "h8" in bishop_moves("a1")


# --------------------------------------------------------------------------- #
# Pawn (written from white's point of view: towards rank 8)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_pawn_never_changes_file(coord):
    for destination in pawn_moves(coord):
        assert destination[0] == coord[0]


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_a_pawn_only_steps_towards_rank_eight(coord):
    for destination in pawn_moves(coord):
        assert int(destination[1]) > int(coord[1])


@pytest.mark.parametrize("file_", FILES)
def test_a_pawn_on_its_starting_rank_may_push_one_or_two(file_):
    assert pawn_moves(file_ + "2") == [file_ + "3", file_ + "4"]


@pytest.mark.parametrize("rank", ["3", "4", "5", "6", "7"])
@pytest.mark.parametrize("file_", FILES)
def test_a_pawn_past_its_starting_rank_may_push_one(file_, rank):
    assert pawn_moves(file_ + rank) == [file_ + str(int(rank) + 1)]


@pytest.mark.parametrize("file_", FILES)
def test_a_pawn_on_the_far_rank_has_nowhere_to_go(file_):
    assert pawn_moves(file_ + "8") == []


@pytest.mark.parametrize("file_", FILES)
def test_the_double_push_is_only_offered_from_rank_two(file_):
    for rank in "134567":
        assert len(pawn_moves(file_ + rank)) <= 1


# --------------------------------------------------------------------------- #
# The generator registry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("piece", "generator"),
    [
        (PAWN, pawn_moves),
        (KNIGHT, knight_moves),
        (BISHOP, bishop_moves),
        (ROOK, rook_moves),
        (QUEEN, queen_moves),
    ],
)
def test_registry_points_each_piece_at_its_generator(piece, generator):
    assert MOVE_FUNCTIONS[piece] is generator


def test_registry_holds_no_entry_for_an_empty_square():
    assert EMPTY_SQUARE not in MOVE_FUNCTIONS


# --------------------------------------------------------------------------- #
# Known gaps - these describe behaviour the module does not have yet.
# Each is strict, so implementing the feature turns the test into a failure
# telling you to drop the marker.
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="king_moves exists but KING was never added to MOVE_FUNCTIONS, "
    "which is what makes visible_positions raise KeyError(6)",
)
def test_the_registry_covers_every_piece_on_the_board():
    assert MOVE_FUNCTIONS[KING] is king_moves


@pytest.mark.xfail(strict=True, reason="pawn_moves takes no colour; it only moves up")
def test_a_black_pawn_steps_towards_rank_one():
    assert "e6" in pawn_moves("e7")


@pytest.mark.xfail(strict=True, reason="pawn_moves generates no diagonal captures")
def test_a_pawn_can_capture_diagonally():
    assert {"d5", "f5"}.issubset(pawn_moves("e4"))
