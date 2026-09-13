'''``gen_pawn_moves`` - the piece with the most special cases.

A pawn moves in one direction only, captures in a direction it cannot move in,
moves two squares only from its starting rank and only over an empty square,
captures a square it never stepped on when taking en passant, and stops being a
pawn on the last rank.  Each of those is tested on its own, and then the whole
lot is swept across midgame positions.

Direction is the thing to keep straight: ``PAWN_STEP = {WHITE: -1, BLACK: 1}``
moves along the *row*, which is index 0 of a ``(row, column)`` coordinate, since
row 0 is rank 8.  A white pawn on e2 is at ``(6, 4)`` and steps to ``(5, 4)``.
'''

import chess
import pytest

from src import game
from tests import oracle

WHITE_PAWN_HOME = '7k/8/8/8/8/8/4P3/K7 w - - 0 1'
BLACK_PAWN_HOME = '7k/4p3/8/8/8/8/8/K7 b - - 0 1'

# A black knight directly in front of the pawn on e2 - no push at all, and no
# capture either, because a pawn cannot take what is straight ahead of it.
PAWN_FACING_PIECE = '7k/8/8/8/8/4n3/4P3/K7 w - - 0 1'

# The square in front is free but e4 is not, so the double push is unavailable.
PAWN_DOUBLE_BLOCKED = '7k/8/8/8/4n3/8/4P3/K7 w - - 0 1'

# Black knights on both of the e2 pawn's capture squares.
PAWN_WITH_CAPTURES = '7k/8/8/8/8/3n1n2/4P3/K7 w - - 0 1'
BLACK_PAWN_WITH_CAPTURES = '7k/4p3/3N1N2/8/8/8/8/K7 b - - 0 1'

# A white pawn already past its home rank - one step only.
WHITE_PAWN_ADVANCED = '7k/8/8/8/8/4P3/8/K7 w - - 0 1'

# Black has just played f7-f5; white's e5 pawn may take it en passant on f6.
WHITE_EN_PASSANT = '7k/8/8/4Pp2/8/8/8/K7 w - f6 0 1'

# White has just played f2-f4; black's e4 pawn may take it en passant on f3.
BLACK_EN_PASSANT = '7k/8/8/8/4pP2/8/8/K7 b - f3 0 1'

PAWN_PROMOTING = '7k/4P3/8/8/8/8/8/K7 w - - 0 1'
PAWN_PROMOTING_WITH_CAPTURE = '3r3k/4P3/8/8/8/8/8/K7 w - - 0 1'


def targets(board, origin, player):
    '''``gen_pawn_moves`` output as a set of coordinate pairs.'''
    return {tuple(move) for move in board.gen_pawn_moves(origin, player)}


def test_white_pawn_pushes_up_the_board():
    '''From its home rank a white pawn offers one square and two.'''
    cb, board = oracle.position(WHITE_PAWN_HOME)
    origin = oracle.at('e2')

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.WHITE), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, f'white pawn e2: missing {missing}, unexpected {extra}'


def test_black_pawn_pushes_down_the_board():
    '''A black pawn moves the other way - towards row 7, not row 0.

    Both colours read the same ``PAWN_STEP`` table, so a sign or axis mistake
    shows up here as a pawn that moves backwards or sideways.
    '''
    cb, board = oracle.position(BLACK_PAWN_HOME)
    origin = oracle.at('e7')

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.BLACK), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, f'black pawn e7: missing {missing}, unexpected {extra}'


def test_advanced_pawn_has_no_double_push():
    '''Off its home rank, a pawn moves one square only.'''
    cb, board = oracle.position(WHITE_PAWN_ADVANCED)
    origin = oracle.at('e3')

    got = targets(board, origin, game.WHITE)

    assert got == {oracle.at('e4')}, (
        f'a pawn on e3 should only reach e4, got {oracle.names(got)}'
    )


def test_pawn_cannot_capture_straight_ahead():
    '''A piece directly in front stops the pawn and is not a capture.'''
    cb, board = oracle.position(PAWN_FACING_PIECE)
    origin = oracle.at('e2')

    got = targets(board, origin, game.WHITE)

    assert got == set(), (
        f'a pawn on e2 facing a knight on e3 has no moves, got {oracle.names(got)}'
    )


def test_double_push_cannot_jump_a_piece():
    '''The double push needs both squares empty, not just the landing square.'''
    cb, board = oracle.position(PAWN_DOUBLE_BLOCKED)
    origin = oracle.at('e2')

    got = targets(board, origin, game.WHITE)

    assert got == {oracle.at('e3')}, (
        f'with a knight on e4 the pawn reaches e3 only, got {oracle.names(got)}'
    )


def test_white_pawn_captures_diagonally():
    '''Both forward diagonals are available when an enemy piece stands there.'''
    cb, board = oracle.position(PAWN_WITH_CAPTURES)
    origin = oracle.at('e2')

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.WHITE), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, (
        f'white pawn e2 with captures on d3/f3: missing {missing}, unexpected {extra}'
    )


def test_black_pawn_captures_diagonally():
    '''A black pawn captures white pieces, on the diagonals ahead of *it*.

    The capture test reads the piece code on the target square and compares it
    against a colour.  Comparing against a fixed colour works for one side and
    quietly fails for the other: black would refuse every capture and offer to
    take its own pieces instead.
    '''
    cb, board = oracle.position(BLACK_PAWN_WITH_CAPTURES)
    origin = oracle.at('e7')

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.BLACK), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, (
        f'black pawn e7 with captures on d6/f6: missing {missing}, unexpected {extra}'
    )


def test_pawn_does_not_capture_its_own_piece(start_board):
    '''In the opening position no pawn has a capture, friendly or otherwise.'''
    for file_ in 'abcdefgh':
        origin = oracle.at(f'{file_}2')
        got = targets(start_board, origin, game.WHITE)
        diagonals = {t for t in got if t[1] != origin[1]}

        assert not diagonals, (
            f'{file_}2 has no captures at the start, got {oracle.names(diagonals)}'
        )


def test_white_captures_en_passant():
    '''White's e5 pawn may take a pawn that just ran past it to f5, landing on f6.

    The captured pawn is on f5 and the pawn lands on f6 - one row further along
    white's direction of travel, same column.  Getting the axis wrong here
    produces a move to e5 or g5, which is not a chess move at all.
    '''
    cb, board = oracle.position(WHITE_EN_PASSANT)
    origin = oracle.at('e5')

    assert board.en_passant_white == oracle.at('f5'), (
        'the fixture should record the black pawn white may capture'
    )

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.WHITE), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, (
        f'white en passant: missing {missing}, unexpected {extra}'
    )


def test_black_captures_en_passant():
    '''Black's e4 pawn may take a pawn that just ran past it to f4, landing on f3.'''
    cb, board = oracle.position(BLACK_EN_PASSANT)
    origin = oracle.at('e4')

    assert board.en_passant_black == oracle.at('f4'), (
        'the fixture should record the white pawn black may capture'
    )

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.BLACK), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, (
        f'black en passant: missing {missing}, unexpected {extra}'
    )


def test_no_en_passant_without_the_right():
    '''With no pawn having just double-pushed, there is no en passant move.'''
    cb, board = oracle.position('7k/8/8/4Pp2/8/8/8/K7 w - - 0 1')
    origin = oracle.at('e5')

    got = targets(board, origin, game.WHITE)

    assert got == {oracle.at('e6')}, (
        f'without an en passant right the e5 pawn only pushes to e6, got {oracle.names(got)}'
    )


def test_en_passant_only_for_an_adjacent_pawn():
    '''A pawn two files away from the double-pushed pawn gets nothing extra.'''
    cb, board = oracle.position('7k/8/8/2P2p2/8/8/8/K7 w - f6 0 1')
    origin = oracle.at('c5')

    got = targets(board, origin, game.WHITE)

    assert got == {oracle.at('c6')}, (
        f'c5 is not adjacent to f5, so only c6 is available, got {oracle.names(got)}'
    )


def test_pawn_on_the_seventh_reaches_the_last_rank():
    '''A pawn one step from promoting still offers that step.

    The push reads the square in front without a bounds check first; on the last
    rank a negative row index wraps silently to the far side of the array rather
    than raising, so this also pins down that the square offered is e8.
    '''
    cb, board = oracle.position(PAWN_PROMOTING)
    origin = oracle.at('e7')

    got = targets(board, origin, game.WHITE)

    assert got == {oracle.at('e8')}, (
        f'a pawn on e7 promotes on e8, got {oracle.names(got)}'
    )


def test_promoting_pawn_can_capture_onto_the_last_rank():
    '''Promotion by capture is offered like any other pawn capture.'''
    cb, board = oracle.position(PAWN_PROMOTING_WITH_CAPTURE)
    origin = oracle.at('e7')

    missing, extra = oracle.diff(
        board.gen_pawn_moves(origin, game.WHITE), oracle.pseudo_legal_targets(cb, origin)
    )

    assert not missing and not extra, (
        f'promotion with capture: missing {missing}, unexpected {extra}'
    )


def test_pawns_agree_across_midgame_positions(fen):
    '''Every pawn of the side to move, in a real position, matches python-chess.

    Only the side to move is checked, because an en passant right belongs to
    whoever is on move - asking about the other side's pawns would compare
    against an oracle that has correctly forgotten the right.
    '''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]

    for origin in oracle.squares_of(cb, player, game.PAWN):
        missing, extra = oracle.diff(
            board.gen_pawn_moves(origin, player),
            oracle.pseudo_legal_targets(cb, origin),
        )
        assert not missing and not extra, (
            f'{fen}: pawn {oracle.name(origin)} - missing {missing}, unexpected {extra}'
        )
