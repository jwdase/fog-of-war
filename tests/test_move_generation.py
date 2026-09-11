'''The move generators that board states are advanced and revealed with.

A generator takes a ``(row, column)`` coordinate - a value of ``COORDINATES`` -
and returns the coordinates it can move to, so an answer can be fed straight
back into another generator or used to index the board array.

Given an occupancy map, ``{coordinate: colour}``, a generator honours it: a line
stops at the first piece in the way, an enemy piece can be taken and a friendly
one cannot, and a pawn pushes the way its colour faces. Called with no board at
all they describe an empty one, which is what the geometry sections check.
'''

import pytest

from src.data.data import (
    BISHOP,
    BLACK,
    COORDINATES,
    EMPTY_SQUARE,
    KING,
    KNIGHT,
    MOVE_FUNCTIONS,
    ON_BOARD,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    bishop_moves,
    king_moves,
    knight_moves,
    occupancy,
    pawn_moves,
    queen_moves,
    rook_moves,
)

from .helpers import ALL_COORDS, ALL_SQUARES, FILES, coord_set, coords

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
OFF_THE_BOARD = [(-1, 0), (0, -1), (8, 0), (0, 8), (9, 9)]


def offsets(origin, destination):
    '''Row/column delta between two coordinates.'''
    return destination[0] - origin[0], destination[1] - origin[1]


# --------------------------------------------------------------------------- #
# Invariants every generator has to satisfy, from every square
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_moves_stay_on_the_board(_name, generator, coord):
    for destination in generator(coord):
        assert destination in ON_BOARD


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_moves_are_coordinates_a_generator_can_be_called_with_again(_name, generator, coord):
    '''Coordinates in, coordinates out - the two ends of the API agree.'''
    for destination in generator(coord):
        assert isinstance(destination, tuple)
        assert generator(destination) is not None


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_moves_are_not_repeated(_name, generator, coord):
    moves = generator(coord)
    assert len(moves) == len(set(moves))


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_piece_never_moves_to_its_own_square(_name, generator, coord):
    assert coord not in generator(coord)


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_generators_are_pure(_name, generator, coord):
    '''Calling a generator twice gives the same answer and mutates no state.'''
    assert generator(coord) == generator(coord)


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_generator_does_not_mutate_the_board_it_is_given(_name, generator, coord):
    board = occupancy({PAWN: ["d4", "e4"]}, {PAWN: ["d5", "e5"]})
    before = dict(board)
    generator(coord, board, WHITE)
    assert board == before


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
@pytest.mark.parametrize("coord", OFF_THE_BOARD)
def test_a_coordinate_off_the_board_is_rejected(_name, generator, coord):
    with pytest.raises(KeyError):
        generator(coord)


# --------------------------------------------------------------------------- #
# Knight
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_knight_moves_are_all_l_shaped(coord):
    for destination in knight_moves(coord):
        assert sorted(abs(d) for d in offsets(coord, destination)) == [1, 2]


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("d4", {"c2", "e2", "b3", "f3", "b5", "f5", "c6", "e6"}),
        ("b1", {"a3", "c3", "d2"}),
        ("g8", {"e7", "f6", "h6"}),
        ("a1", {"b3", "c2"}),
        ("h8", {"f7", "g6"}),
    ],
)
def test_knight_move_sets(origin, expected):
    assert set(knight_moves(COORDINATES[origin])) == coord_set(expected)


@pytest.mark.parametrize("coord", coords(CENTRE), ids=CENTRE)
def test_a_central_knight_has_eight_moves(coord):
    assert len(knight_moves(coord)) == 8


@pytest.mark.parametrize("coord", coords(CORNERS), ids=CORNERS)
def test_a_cornered_knight_has_two_moves(coord):
    assert len(knight_moves(coord)) == 2


# --------------------------------------------------------------------------- #
# King
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_king_only_steps_one_square(coord):
    for destination in king_moves(coord):
        dx, dy = offsets(coord, destination)
        assert max(abs(dx), abs(dy)) == 1


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("e1", {"d1", "f1", "d2", "e2", "f2"}),
        ("a8", {"a7", "b7", "b8"}),
        ("d4", {"c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"}),
    ],
)
def test_king_move_sets(origin, expected):
    assert set(king_moves(COORDINATES[origin])) == coord_set(expected)


@pytest.mark.parametrize("coord", coords(CENTRE), ids=CENTRE)
def test_a_central_king_has_eight_moves(coord):
    assert len(king_moves(coord)) == 8


@pytest.mark.parametrize("coord", coords(CORNERS), ids=CORNERS)
def test_a_cornered_king_has_three_moves(coord):
    assert len(king_moves(coord)) == 3


EDGES = ["e1", "e8", "a4", "h4"]


@pytest.mark.parametrize("coord", coords(EDGES), ids=EDGES)
def test_a_king_on_an_edge_has_five_moves(coord):
    assert len(king_moves(coord)) == 5


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_king_reaches_every_neighbouring_square(coord):
    row, col = coord
    neighbours = {
        (row + dx, col + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    }
    assert set(king_moves(coord)) == neighbours & ON_BOARD


# --------------------------------------------------------------------------- #
# Bishop
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_bishop_moves_stay_on_a_diagonal(coord):
    for destination in bishop_moves(coord):
        dx, dy = offsets(coord, destination)
        assert abs(dx) == abs(dy) != 0


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_bishop_keeps_its_square_colour(coord):
    '''Diagonal movement preserves the parity of row + column.'''
    row, col = coord
    for new_row, new_col in bishop_moves(coord):
        assert (new_row + new_col) % 2 == (row + col) % 2


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("a1", {"b2", "c3", "d4", "e5", "f6", "g7", "h8"}),
        ("h1", {"g2", "f3", "e4", "d5", "c6", "b7", "a8"}),
        ("c1", {"b2", "a3", "d2", "e3", "f4", "g5", "h6"}),
    ],
)
def test_bishop_move_sets(origin, expected):
    assert set(bishop_moves(COORDINATES[origin])) == coord_set(expected)


@pytest.mark.parametrize("coord", coords(CENTRE), ids=CENTRE)
def test_a_central_bishop_has_thirteen_moves(coord):
    assert len(bishop_moves(coord)) == 13


@pytest.mark.parametrize("coord", coords(CORNERS), ids=CORNERS)
def test_a_cornered_bishop_sweeps_the_long_diagonal(coord):
    assert len(bishop_moves(coord)) == 7


# --------------------------------------------------------------------------- #
# Rook
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_rook_moves_share_a_rank_or_a_file(coord):
    for destination in rook_moves(coord):
        dx, dy = offsets(coord, destination)
        assert (dx == 0) != (dy == 0)


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_rook_always_has_fourteen_moves(coord):
    assert len(rook_moves(coord)) == 14


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_rook_reaches_its_whole_rank_and_file(coord):
    row, col = coord
    whole_cross = {(row, c) for c in range(8)} | {(r, col) for r in range(8)}
    assert set(rook_moves(coord)) == whole_cross - {coord}


# --------------------------------------------------------------------------- #
# Queen
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_queen_combines_bishop_and_rook(coord):
    assert set(queen_moves(coord)) == set(bishop_moves(coord)) | set(rook_moves(coord))


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_bishop_and_rook_reach_disjoint_squares(coord):
    '''Which is why concatenating them cannot introduce a duplicate.'''
    assert set(bishop_moves(coord)).isdisjoint(set(rook_moves(coord)))
    assert len(queen_moves(coord)) == len(bishop_moves(coord)) + len(rook_moves(coord))


@pytest.mark.parametrize("coord", coords(CENTRE), ids=CENTRE)
def test_a_central_queen_has_twenty_seven_moves(coord):
    assert len(queen_moves(coord)) == 27


@pytest.mark.parametrize("coord", coords(CORNERS), ids=CORNERS)
def test_a_cornered_queen_has_twenty_one_moves(coord):
    assert len(queen_moves(coord)) == 21


# --------------------------------------------------------------------------- #
# Reciprocity on an empty board
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("_name", "generator"), SLIDERS + STEPPERS, ids=["bishop", "rook", "queen", "knight", "king"]
)
@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_moves_are_reciprocal(_name, generator, coord):
    '''Every generator except the pawn is symmetric: if a can reach b, b can reach a.'''
    for destination in generator(coord):
        assert coord in generator(destination)


# --------------------------------------------------------------------------- #
# The board the piece stands on
#
# ``occupancy`` builds the map the generators read from the same
# ``{piece: [square, ...]}`` dictionaries a ChessState holds.
# --------------------------------------------------------------------------- #

def test_an_empty_board_and_no_board_are_the_same_thing():
    assert rook_moves(COORDINATES["a1"], {}, WHITE) == rook_moves(COORDINATES["a1"])


def test_a_line_stops_in_front_of_a_friendly_piece():
    board = occupancy({ROOK: ["a1"], PAWN: ["a4"]}, {})
    assert set(rook_moves(COORDINATES["a1"], board, WHITE)) == coord_set(
        ["a2", "a3"] + [f + "1" for f in "bcdefgh"]
    )


def test_a_line_stops_on_an_enemy_piece_and_takes_it():
    board = occupancy({ROOK: ["a1"]}, {PAWN: ["a4"]})
    moves = set(rook_moves(COORDINATES["a1"], board, WHITE))
    assert COORDINATES["a4"] in moves
    assert COORDINATES["a5"] not in moves


def test_the_same_board_reads_differently_to_each_colour():
    '''a4 is a capture for the white rook on a1 and a friend to a black one.'''
    board = occupancy({ROOK: ["a1"]}, {PAWN: ["a4"]})
    assert COORDINATES["a4"] in rook_moves(COORDINATES["a1"], board, WHITE)
    assert COORDINATES["a4"] not in rook_moves(COORDINATES["a1"], board, BLACK)


def test_a_bishop_is_shut_in_by_its_own_pawns():
    board = occupancy({BISHOP: ["c1"], PAWN: ["b2", "d2"]}, {})
    assert bishop_moves(COORDINATES["c1"], board, WHITE) == []


def test_a_queen_still_combines_bishop_and_rook_on_a_crowded_board():
    board = occupancy({QUEEN: ["d1"], PAWN: ["c2", "d2"]}, {ROOK: ["a1"], BISHOP: ["f3"]})
    coord = COORDINATES["d1"]
    assert set(queen_moves(coord, board, WHITE)) == set(
        bishop_moves(coord, board, WHITE)
    ) | set(rook_moves(coord, board, WHITE))


def test_a_knight_jumps_over_the_pieces_in_its_way():
    '''Only the landing square can turn a knight's move away.'''
    board = occupancy({KNIGHT: ["b1"], PAWN: ["a2", "b2", "c2"]}, {})
    assert set(knight_moves(COORDINATES["b1"], board, WHITE)) == coord_set(["a3", "c3", "d2"])


@pytest.mark.parametrize(
    ("_name", "generator", "origin", "landing"),
    [
        ("knight", knight_moves, "b1", "d2"),
        ("king", king_moves, "e1", "e2"),
    ],
    ids=["knight", "king"],
)
def test_a_stepper_may_take_an_enemy_but_not_land_on_a_friend(_name, generator, origin, landing):
    friend = occupancy({PAWN: [landing]}, {})
    enemy = occupancy({}, {PAWN: [landing]})
    assert COORDINATES[landing] not in generator(COORDINATES[origin], friend, WHITE)
    assert COORDINATES[landing] in generator(COORDINATES[origin], enemy, WHITE)


# --------------------------------------------------------------------------- #
# Pawn
#
# The one piece that is not symmetric and does not move the way it captures.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_pawn_with_nothing_to_take_never_changes_file(coord):
    for destination in pawn_moves(coord):
        assert destination[1] == coord[1]


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_white_pawn_only_steps_towards_rank_eight(coord):
    '''Rank 8 is row 0, so white's pawns walk towards a smaller row.'''
    for destination in pawn_moves(coord, {}, WHITE):
        assert destination[0] < coord[0]


@pytest.mark.parametrize("coord", ALL_COORDS, ids=ALL_SQUARES)
def test_a_black_pawn_only_steps_towards_rank_one(coord):
    for destination in pawn_moves(coord, {}, BLACK):
        assert destination[0] > coord[0]


@pytest.mark.parametrize("file_", FILES)
def test_a_white_pawn_on_its_starting_rank_may_push_one_or_two(file_):
    assert pawn_moves(COORDINATES[file_ + "2"]) == coords([file_ + "3", file_ + "4"])


@pytest.mark.parametrize("file_", FILES)
def test_a_black_pawn_on_its_starting_rank_may_push_one_or_two(file_):
    assert pawn_moves(COORDINATES[file_ + "7"], {}, BLACK) == coords(
        [file_ + "6", file_ + "5"]
    )


@pytest.mark.parametrize("rank", ["3", "4", "5", "6", "7"])
@pytest.mark.parametrize("file_", FILES)
def test_a_pawn_past_its_starting_rank_may_push_one(file_, rank):
    assert pawn_moves(COORDINATES[file_ + rank]) == coords([file_ + str(int(rank) + 1)])


@pytest.mark.parametrize("file_", FILES)
def test_a_pawn_on_the_far_rank_has_nowhere_to_go(file_):
    '''Promotion is not modelled: a pawn that reaches rank 8 simply stops.'''
    assert pawn_moves(COORDINATES[file_ + "8"]) == []


@pytest.mark.parametrize("file_", FILES)
def test_the_double_push_is_only_offered_from_rank_two(file_):
    for rank in "134567":
        assert len(pawn_moves(COORDINATES[file_ + rank])) <= 1


def test_a_pawn_cannot_push_onto_an_occupied_square():
    for board in (occupancy({PAWN: ["e4", "e5"]}, {}), occupancy({PAWN: ["e4"]}, {PAWN: ["e5"]})):
        assert pawn_moves(COORDINATES["e4"], board, WHITE) == []


def test_the_double_push_needs_both_squares_clear():
    blocked_near = occupancy({PAWN: ["e2"]}, {PAWN: ["e3"]})
    blocked_far = occupancy({PAWN: ["e2"]}, {PAWN: ["e4"]})
    assert pawn_moves(COORDINATES["e2"], blocked_near, WHITE) == []
    assert pawn_moves(COORDINATES["e2"], blocked_far, WHITE) == coords(["e3"])


def test_a_pawn_captures_diagonally():
    board = occupancy({PAWN: ["e4"]}, {PAWN: ["d5", "f5"]})
    assert set(pawn_moves(COORDINATES["e4"], board, WHITE)) == coord_set(["e5", "d5", "f5"])


def test_a_pawn_does_not_capture_its_own_pieces():
    board = occupancy({PAWN: ["e4", "d5", "f5"]}, {})
    assert pawn_moves(COORDINATES["e4"], board, WHITE) == coords(["e5"])


def test_a_pawn_does_not_capture_an_empty_diagonal():
    assert set(pawn_moves(COORDINATES["e4"], {}, WHITE)) == coord_set(["e5"])


def test_a_pawn_may_take_the_en_passant_square():
    '''The one capture onto an empty square, so the square has to be named.'''
    board = occupancy({PAWN: ["e5"]}, {PAWN: ["d5"]})
    coord = COORDINATES["e5"]
    assert COORDINATES["d6"] not in pawn_moves(coord, board, WHITE)
    assert COORDINATES["d6"] in pawn_moves(coord, board, WHITE, en_passant=COORDINATES["d6"])


def test_black_pawns_capture_towards_rank_one():
    board = occupancy({PAWN: ["d4", "f4"]}, {PAWN: ["e5"]})
    assert set(pawn_moves(COORDINATES["e5"], board, BLACK)) == coord_set(["e4", "d4", "f4"])


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
        (KING, king_moves),
    ],
)
def test_registry_points_each_piece_at_its_generator(piece, generator):
    assert MOVE_FUNCTIONS[piece] is generator


def test_the_registry_covers_every_piece_on_the_board():
    '''Anything that reads MOVE_FUNCTIONS off a board array needs all six.'''
    assert set(MOVE_FUNCTIONS) == {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}


def test_registry_holds_no_entry_for_an_empty_square():
    assert EMPTY_SQUARE not in MOVE_FUNCTIONS


@pytest.mark.parametrize(("_name", "generator"), GENERATORS, ids=GENERATOR_IDS)
def test_every_generator_takes_the_same_arguments(_name, generator):
    '''So a caller can dispatch through the registry without special cases.'''
    board = occupancy({PAWN: ["d4"]}, {PAWN: ["e5"]})
    assert generator(COORDINATES["d4"], board, WHITE) is not None


# --------------------------------------------------------------------------- #
# Known gaps - what a generator still does not know about
# --------------------------------------------------------------------------- #

def test_castling_is_not_a_king_move():
    '''0-0 is written as its own move, so ChessGame.make_move has to resolve it.'''
    board = occupancy({KING: ["e1"], ROOK: ["a1", "h1"]}, {})
    moves = king_moves(COORDINATES["e1"], board, WHITE)
    assert COORDINATES["g1"] not in moves
    assert COORDINATES["c1"] not in moves


def test_moves_are_pseudo_legal():
    '''Nothing here knows about check, so a pinned piece is still offered its
    moves and a king is still offered a square it would be captured on.'''
    board = occupancy({KING: ["e1"], BISHOP: ["e2"]}, {ROOK: ["e8"]})
    assert bishop_moves(COORDINATES["e2"], board, WHITE) != []
    assert COORDINATES["d1"] in king_moves(COORDINATES["e1"], board, WHITE)
