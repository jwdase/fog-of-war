'''``gen_king_moves`` - the one generator that claims to produce *legal* moves.

Every other generator ignores check.  This one does not: it builds the set of
squares the opponent attacks and refuses to step onto them.  So this is the one
generator measured against python-chess's ``legal_moves`` rather than its
pseudo-legal moves.

Castling is checked separately at the bottom, because it is a king move in the
rules and a king move in python-chess's legal move list.
'''

import chess
import pytest

from src import game
from tests import oracle

LONE_WHITE_KING = '7k/8/8/8/3K4/8/8/8 w - - 0 1'
CORNER_WHITE_KING = '7k/8/8/8/8/8/8/K7 w - - 0 1'

# A black rook on e8 owns the e-file, so the white king on d4 may not step onto
# e3, e4 or e5.
KING_NEXT_TO_A_FILE = '4r2k/8/8/8/3K4/8/8/8 w - - 0 1'

# An undefended black pawn on d5, right next to the white king.
KING_WITH_A_CAPTURE = '7k/8/8/3p4/3K4/8/8/8 w - - 0 1'

# The same, but the pawn is defended by a rook on d8 - taking it would be moving
# into check.
KING_WITH_A_DEFENDED_PAWN = '3r3k/8/8/3p4/3K4/8/8/8 w - - 0 1'

# Black to move: the white rook on e3 owns the e-file, so the black king on d5
# may not step onto e4, e5 or e6.
BLACK_KING_IN_TROUBLE = '8/8/8/3k4/8/4R3/8/K7 b - - 0 1'

CASTLING_AVAILABLE = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'


def targets(board, origin, player):
    '''``gen_king_moves`` output as a set of coordinate pairs.'''
    return {tuple(move) for move in board.gen_king_moves(origin, player)}


@pytest.mark.parametrize(
    'fen_, origin',
    [(LONE_WHITE_KING, 'd4'), (CORNER_WHITE_KING, 'a1')],
    ids=['centre', 'corner'],
)
def test_king_steps_one_square_in_every_direction(fen_, origin):
    '''An unthreatened king reaches its neighbours, clipped at the board edge.'''
    cb, board = oracle.position(fen_)
    coord = oracle.at(origin)

    missing, extra = oracle.diff(
        board.gen_king_moves(coord, game.WHITE),
        oracle.legal_targets(cb, coord, castling=False),
    )

    assert not missing and not extra, (
        f'king {origin}: missing {missing}, unexpected {extra}'
    )


def test_king_may_capture_an_undefended_piece():
    '''A king takes an enemy piece next to it when nothing defends it.

    The generator keeps only squares that read as empty, which rules out every
    capture the king has - a king that can never take anything is not playing
    chess.
    '''
    cb, board = oracle.position(KING_WITH_A_CAPTURE)
    origin = oracle.at('d4')

    got = targets(board, origin, game.WHITE)

    assert oracle.at('d5') in got, (
        f'the king on d4 should be able to take the pawn on d5, got {oracle.names(got)}'
    )


def test_king_may_not_capture_a_defended_piece():
    '''Taking a defended piece would put the king in check, so it is not offered.'''
    cb, board = oracle.position(KING_WITH_A_DEFENDED_PAWN)
    origin = oracle.at('d4')

    got = targets(board, origin, game.WHITE)

    assert oracle.at('d5') not in got, (
        'the pawn on d5 is defended by the rook on d8, so Kxd5 is illegal'
    )


def test_king_may_not_step_into_check():
    '''Squares the opponent attacks are excluded, even though they are empty.'''
    cb, board = oracle.position(KING_NEXT_TO_A_FILE)
    origin = oracle.at('d4')

    got = targets(board, origin, game.WHITE)

    for forbidden in ('e3', 'e4', 'e5'):
        assert oracle.at(forbidden) not in got, (
            f'{forbidden} is covered by the rook on e8'
        )
    for allowed in ('c3', 'c4', 'c5', 'd3', 'd5'):
        assert oracle.at(allowed) in got, f'{allowed} is safe and should be offered'


def test_white_king_avoids_squares_black_attacks():
    '''The full move list for a threatened white king matches python-chess.'''
    cb, board = oracle.position(KING_NEXT_TO_A_FILE)
    origin = oracle.at('d4')

    missing, extra = oracle.diff(
        board.gen_king_moves(origin, game.WHITE),
        oracle.legal_targets(cb, origin, castling=False),
    )

    assert not missing and not extra, (
        f'white king d4 vs rook e8: missing {missing}, unexpected {extra}'
    )


def test_black_king_avoids_squares_white_attacks():
    '''A black king consults *white's* attacks, not black's own.

    The opponent has to be worked out from the player, and a colour expression
    that collapses to one side leaves one of the two kings walking into check
    while the other refuses to move next to its own pieces.
    '''
    cb, board = oracle.position(BLACK_KING_IN_TROUBLE)
    origin = oracle.at('d5')

    got = targets(board, origin, game.BLACK)

    for forbidden in ('e4', 'e5', 'e6'):
        assert oracle.at(forbidden) not in got, (
            f'{forbidden} is covered by the white rook on e3'
        )

    missing, extra = oracle.diff(
        got, oracle.legal_targets(cb, origin, castling=False)
    )
    assert not missing and not extra, (
        f'black king d5 vs rook e3: missing {missing}, unexpected {extra}'
    )


def test_king_may_castle_both_ways():
    '''With rights intact and the path clear, castling is among the king's moves.

    python-chess reports castling as a king move from e1 to g1 or c1, so a legal
    move generator has to offer those squares.
    '''
    cb, board = oracle.position(CASTLING_AVAILABLE)
    origin = oracle.at('e1')

    got = targets(board, origin, game.WHITE)

    assert oracle.at('g1') in got, 'kingside castling should be available'
    assert oracle.at('c1') in got, 'queenside castling should be available'


def test_kings_agree_across_midgame_positions(fen):
    '''The king of the side to move, in a real position, matches python-chess.

    Castling is excluded from both sides of the comparison - a two-square king
    move is dropped from the generator's output as well as from the oracle's - so
    that a missing castling move fails only the test above, and this one stays
    about ordinary king steps.
    '''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]

    for origin in oracle.squares_of(cb, player, game.KING):
        steps = [
            target
            for target in board.gen_king_moves(origin, player)
            if abs(target[1] - origin[1]) <= 1
        ]
        missing, extra = oracle.diff(
            steps, oracle.legal_targets(cb, origin, castling=False)
        )
        assert not missing and not extra, (
            f'{fen}: king {oracle.name(origin)} - missing {missing}, unexpected {extra}'
        )
