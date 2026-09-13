'''``gen_moves_knight`` - the leaper.

A knight's rules are short: eight fixed offsets, keep the ones on the board,
drop the ones standing on your own pieces, and ignore everything in between
because a knight jumps.

Like ``gen_pawn_moves`` and ``gen_king_moves``, the generator is told whose
knight it is - it needs the colour to know which landing squares are its own.
The attack-map counterpart, ``gen_checks_knight``, does not: a knight covers
all eight squares whatever stands on them.  ``test_attacked_squares`` has that.
'''

import pytest

from src import game
from tests import oracle

CENTRE_KNIGHT = '7k/8/8/8/3N4/8/8/K7 w - - 0 1'
CORNER_KNIGHT = '7k/8/8/8/8/8/8/N6K w - - 0 1'
EDGE_KNIGHT = '7k/8/8/7N/8/8/8/K7 w - - 0 1'

# A knight on d4 hemmed in by its own pawns on some landing squares and black
# pawns on others.
CROWDED_KNIGHT = '7k/8/2p1p3/8/3N4/8/2P1P3/K7 w - - 0 1'


@pytest.mark.parametrize(
    'fen_, origin',
    [
        (CENTRE_KNIGHT, 'd4'),
        (CORNER_KNIGHT, 'a1'),
        (EDGE_KNIGHT, 'h5'),
    ],
    ids=['centre', 'corner', 'edge'],
)
def test_knight_stays_on_the_board(fen_, origin):
    '''From the middle, a corner and an edge, the knight's eight offsets get clipped.'''
    cb, board = oracle.position(fen_)
    coord = oracle.at(origin)

    missing, extra = oracle.diff(
        board.gen_moves_knight(coord, game.WHITE),
        oracle.pseudo_legal_targets(cb, coord),
    )

    assert not missing and not extra, (
        f'knight {origin}: missing {missing}, unexpected {extra}'
    )


def test_knight_offsets_are_all_l_shaped():
    '''Every target is two squares one way and one the other.

    Unpacking ``KNIGHT_STEPS`` wrongly is the easy mistake here, and a wrong
    unpacking still produces eight-ish plausible squares.
    '''
    _, board = oracle.position(CENTRE_KNIGHT)
    row, col = oracle.at('d4')

    targets = board.gen_moves_knight((row, col), game.WHITE)
    assert len(set(map(tuple, targets))) == 8, (
        f'an open d4 knight has 8 moves, got {oracle.names(targets)}'
    )

    for target in targets:
        shape = sorted((abs(target[0] - row), abs(target[1] - col)))
        assert shape == [1, 2], f'{oracle.name(target)} is not a knight move from d4'


def test_knight_jumps_over_pieces(start):
    '''A knight in the opening position is not blocked by the pawns in front of it.'''
    cb, board = start
    origin = oracle.at('b1')

    missing, extra = oracle.diff(
        board.gen_moves_knight(origin, game.WHITE),
        oracle.pseudo_legal_targets(cb, origin),
    )

    assert not missing and not extra, (
        f'knight b1 at the start: missing {missing}, unexpected {extra}'
    )


def test_knight_may_not_land_on_its_own_piece():
    '''Friendly pieces on landing squares are dropped; enemy pieces are kept.'''
    cb, board = oracle.position(CROWDED_KNIGHT)
    origin = oracle.at('d4')

    targets = {tuple(move) for move in board.gen_moves_knight(origin, game.WHITE)}

    assert oracle.at('c6') in targets, 'the black pawn on c6 is capturable'
    assert oracle.at('e6') in targets, 'the black pawn on e6 is capturable'
    assert oracle.at('c2') not in targets, "c2 holds white's own pawn"
    assert oracle.at('e2') not in targets, "e2 holds white's own pawn"


def test_knights_agree_across_midgame_positions(fen):
    '''Every knight on the board, in a real position, matches python-chess.'''
    cb, board = oracle.position(fen)

    for player in (game.WHITE, game.BLACK):
        turned = oracle.to_move(cb, player)

        for origin in oracle.squares_of(cb, player, game.KNIGHT):
            missing, extra = oracle.diff(
                board.gen_moves_knight(origin, player),
                oracle.pseudo_legal_targets(turned, origin),
            )
            assert not missing and not extra, (
                f'{fen}: knight {oracle.name(origin)} - '
                f'missing {missing}, unexpected {extra}'
            )
