'''``_slide_expand_move``, ``gen_moves_rook``, ``gen_moves_bishop``, ``gen_moves_queen``.

The sliding pieces share one walker, so they share one set of rules: travel in a
fixed direction until the edge of the board, stop on the first piece in the way,
and include that square only if the piece standing there belongs to the
opponent.  The oracle is python-chess's pseudo-legal move list, which is exactly
that - the generators here do no self-check filtering, so they are not held to
it.

The walker comes in two halves.  ``_slide_expand_move`` is the one tested here;
``_slide_expand_check`` answers the attack-map question instead, keeping the
blocking square whoever stands on it, and ``test_attacked_squares`` holds it to
that.  Each move generator is told whose piece it is, since only the colour
distinguishes a capture from a friendly blocker.
'''

import chess
import pytest

from src import game
from tests import oracle

LONE_ROOK = '7k/8/8/8/3R4/8/8/K7 w - - 0 1'
LONE_BISHOP = '7k/8/8/8/3B4/8/8/K7 w - - 0 1'
LONE_QUEEN = '7k/8/8/8/3Q4/8/8/K7 w - - 0 1'

# A white rook on d4 with a black pawn ahead of it on d5 and its own pawn
# behind it on d2: one direction ends in a capture, one ends in a wall.
BOXED_ROOK = '7k/8/8/3p4/3R4/8/3P4/K7 w - - 0 1'


def test_slide_expand_returns_coordinate_pairs():
    '''The walker hands back ``(row, column)`` pairs, not loose numbers.

    Everything downstream - ``check_squares``, ``gen_king_moves``, the callers
    that index the array with these values - treats each entry as a coordinate
    pair.
    '''
    _, board = oracle.position(LONE_ROOK)

    moves = board._slide_expand_move(
        game.STRAIGHT_STEPS, oracle.at('d4'), game.WHITE
    )

    assert moves, 'a rook on an open d4 has somewhere to go'
    for move in moves:
        assert isinstance(move, tuple) and len(move) == 2, (
            f'expected a (row, column) pair, got {move!r}'
        )


def test_rook_on_an_open_board():
    '''An unobstructed rook reaches every square on its rank and file.'''
    cb, board = oracle.position(LONE_ROOK)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.gen_moves_rook(origin, game.WHITE),
        oracle.pseudo_legal_targets(cb, origin),
    )

    assert not missing and not extra, f'rook d4: missing {missing}, unexpected {extra}'


def test_rook_moves_in_straight_lines():
    '''A rook's targets share a rank or a file with it - never a diagonal.

    ``gen_moves_rook`` and ``gen_moves_bishop`` are one swapped argument away
    from each other, and a swap produces a full, plausible-looking move list.
    '''
    _, board = oracle.position(LONE_ROOK)
    row, col = oracle.at('d4')

    for target in board.gen_moves_rook((row, col), game.WHITE):
        assert target[0] == row or target[1] == col, (
            f'{oracle.name(target)} is a diagonal move, not a rook move'
        )


def test_bishop_moves_on_diagonals():
    '''A bishop's targets are all an equal number of rows and columns away.'''
    _, board = oracle.position(LONE_BISHOP)
    row, col = oracle.at('d4')

    for target in board.gen_moves_bishop((row, col), game.WHITE):
        assert abs(target[0] - row) == abs(target[1] - col), (
            f'{oracle.name(target)} is a straight move, not a bishop move'
        )


def test_bishop_on_an_open_board():
    '''An unobstructed bishop reaches both of its diagonals to the edge.'''
    cb, board = oracle.position(LONE_BISHOP)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.gen_moves_bishop(origin, game.WHITE),
        oracle.pseudo_legal_targets(cb, origin),
    )

    assert not missing and not extra, f'bishop d4: missing {missing}, unexpected {extra}'


def test_slider_captures_an_enemy_piece_and_stops_behind_it():
    '''The first enemy piece in a direction is reachable; the square past it is not.'''
    cb, board = oracle.position(BOXED_ROOK)
    origin = oracle.at('d4')

    targets = {tuple(move) for move in board.gen_moves_rook(origin, game.WHITE)}

    assert oracle.at('d5') in targets, 'the black pawn on d5 is capturable'
    assert oracle.at('d6') not in targets, 'd6 is behind the pawn on d5'
    assert oracle.at('d7') not in targets, 'd7 is behind the pawn on d5'


def test_slider_stops_before_its_own_piece():
    '''A friendly piece blocks the direction and is not itself a legal target.

    A walker that stops *after* the blocker rather than before it reports an own
    piece as reachable.  As an attack map that is the correct answer - a defended
    piece is still defended, see ``test_attacked_squares`` - which is why the two
    questions get two walkers.  As a move list it offers the rook its own pawn.
    '''
    cb, board = oracle.position(BOXED_ROOK)
    origin = oracle.at('d4')

    targets = {tuple(move) for move in board.gen_moves_rook(origin, game.WHITE)}

    assert oracle.at('d3') in targets, 'd3 is empty and in front of the blocker'
    assert oracle.at('d2') not in targets, "d2 holds white's own pawn"
    assert oracle.at('d1') not in targets, 'd1 is behind the pawn on d2'


def test_blocked_rook_matches_the_oracle():
    '''With pieces on both sides of it, the rook's move list still agrees.'''
    cb, board = oracle.position(BOXED_ROOK)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.gen_moves_rook(origin, game.WHITE),
        oracle.pseudo_legal_targets(cb, origin),
    )

    assert not missing and not extra, f'boxed rook: missing {missing}, unexpected {extra}'


def test_queen_on_an_open_board():
    '''A queen covers the rook's lines and the bishop's diagonals together.'''
    cb, board = oracle.position(LONE_QUEEN)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.gen_moves_queen(origin, game.WHITE),
        oracle.pseudo_legal_targets(cb, origin),
    )

    assert not missing and not extra, f'queen d4: missing {missing}, unexpected {extra}'


@pytest.mark.parametrize(
    'piece, generator',
    [
        (game.ROOK, 'gen_moves_rook'),
        (game.BISHOP, 'gen_moves_bishop'),
        (game.QUEEN, 'gen_moves_queen'),
    ],
    ids=['rook', 'bishop', 'queen'],
)
def test_sliders_agree_across_midgame_positions(fen, piece, generator):
    '''Every slider on the board, in a real position, matches python-chess.

    This is where the crowded board gets tested: friendly pieces in the way,
    captures on offer, and lines that run off the edge.
    '''
    cb, board = oracle.position(fen)

    for player in (game.WHITE, game.BLACK):
        turned = oracle.to_move(cb, player)

        for origin in oracle.squares_of(cb, player, piece):
            missing, extra = oracle.diff(
                getattr(board, generator)(origin, player),
                oracle.pseudo_legal_targets(turned, origin),
            )
            assert not missing and not extra, (
                f'{fen}: {generator} from {oracle.name(origin)} - '
                f'missing {missing}, unexpected {extra}'
            )
