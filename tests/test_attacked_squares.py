'''``check_squares``, ``get_checks_pawn``, ``get_checks_king`` - the attack map.

These answer a different question from the move generators, and the difference
matters: a square is attacked whether it is empty, holds an enemy piece, or
holds one of your own.  A rook defending its own knight attacks that knight's
square without being able to move there, and that is exactly the information a
king needs in order not to take the knight.

The oracle is python-chess's ``is_attacked_by`` (for a whole side) and
``Board.attacks`` (for one piece), which use those same semantics.
'''

import chess
import pytest

from src import game
from tests import oracle

WHITE_PAWN = '7k/8/8/8/4P3/8/8/K7 w - - 0 1'
BLACK_PAWN = '7k/8/8/4p3/8/8/8/K7 b - - 0 1'
EDGE_PAWN = '7k/8/8/8/P7/8/8/K7 w - - 0 1'

# A white king on d4 with its own pawn on d5: d5 is attacked by the king.
KING_BESIDE_OWN_PAWN = '7k/8/8/3P4/3K4/8/8/8 w - - 0 1'

LONE_QUEEN = '7k/8/8/8/3Q4/8/8/K7 w - - 0 1'


def test_white_pawn_attacks_the_two_squares_ahead_of_it():
    '''A white pawn on e4 attacks d5 and f5 - towards row 0.'''
    cb, board = oracle.position(WHITE_PAWN)
    origin = oracle.at('e4')

    missing, extra = oracle.diff(
        board.get_checks_pawn(origin, game.WHITE), oracle.attacks_from(cb, origin)
    )

    assert not missing and not extra, (
        f'white pawn e4 attacks: missing {missing}, unexpected {extra}'
    )


def test_black_pawn_attacks_the_two_squares_ahead_of_it():
    '''A black pawn on e5 attacks d4 and f4 - the other way from a white pawn.'''
    cb, board = oracle.position(BLACK_PAWN)
    origin = oracle.at('e5')

    missing, extra = oracle.diff(
        board.get_checks_pawn(origin, game.BLACK), oracle.attacks_from(cb, origin)
    )

    assert not missing and not extra, (
        f'black pawn e5 attacks: missing {missing}, unexpected {extra}'
    )


def test_edge_pawn_attacks_one_square_only():
    '''A pawn on the a-file has no attack to its left.'''
    cb, board = oracle.position(EDGE_PAWN)
    origin = oracle.at('a4')

    missing, extra = oracle.diff(
        board.get_checks_pawn(origin, game.WHITE), oracle.attacks_from(cb, origin)
    )

    assert not missing and not extra, (
        f'a4 pawn attacks: missing {missing}, unexpected {extra}'
    )


def test_pawn_attacks_do_not_include_the_push():
    '''The square a pawn moves to is not a square it attacks.'''
    cb, board = oracle.position(WHITE_PAWN)
    origin = oracle.at('e4')

    attacks = {tuple(c) for c in board.get_checks_pawn(origin, game.WHITE)}

    assert oracle.at('e5') not in attacks, 'a pawn does not attack the square in front'


def test_king_attacks_squares_its_own_pieces_stand_on():
    '''A king attacks all eight neighbours, occupied or not.

    Filtering the attack map down to empty squares makes a defended piece look
    safe, which is how an opposing king ends up allowed to capture it.
    '''
    cb, board = oracle.position(KING_BESIDE_OWN_PAWN)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.get_checks_king(origin), oracle.attacks_from(cb, origin)
    )

    assert not missing and not extra, (
        f'king d4 attacks: missing {missing}, unexpected {extra}'
    )


def test_king_attacks_are_clipped_at_the_edge():
    '''A king in the corner attacks three squares.'''
    cb, board = oracle.position('7k/8/8/8/8/8/8/K7 w - - 0 1')
    origin = oracle.at('a1')

    missing, extra = oracle.diff(
        board.get_checks_king(origin), oracle.attacks_from(cb, origin)
    )

    assert not missing and not extra, (
        f'king a1 attacks: missing {missing}, unexpected {extra}'
    )


def test_queens_are_counted_among_the_attackers():
    '''A side's attack map includes what its queen covers.

    ``check_squares`` dispatches on piece code; a piece code with no branch
    contributes nothing, and a queen contributing nothing means a king will
    happily step onto a square the queen is covering.
    '''
    cb, board = oracle.position(LONE_QUEEN)

    covered = {tuple(c) for c in board.check_squares(game.WHITE)}

    for square_ in ('d8', 'a4', 'h4', 'd1', 'a1', 'g7'):
        assert oracle.at(square_) in covered, (
            f'{square_} is covered by the queen on d4 but is missing from check_squares'
        )


def test_attack_map_matches_python_chess(fen):
    '''For both sides, in a real position, the attack map matches python-chess.

    This is the strongest statement in the suite about check detection: if these
    sets agree, then "is this square safe for my king" is being answered
    correctly for every square on the board.
    '''
    cb, board = oracle.position(fen)

    for player, label in ((game.WHITE, 'white'), (game.BLACK, 'black')):
        missing, extra = oracle.diff(
            board.check_squares(player), oracle.attacked_coords(cb, player)
        )
        assert not missing and not extra, (
            f'{fen}: {label} attack map - missing {missing}, unexpected {extra}'
        )
