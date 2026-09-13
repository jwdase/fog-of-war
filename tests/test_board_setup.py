'''``Board.__init__`` - does the constructed board hold the position it was given?

Every later test depends on this, because every later test builds its position
by handing piece dictionaries to the constructor.  If the array does not come
out of the constructor faithfully, a move generator reading that array cannot
possibly agree with python-chess, and the failure would show up somewhere
misleading.
'''

import chess
import pytest

from src import game
from tests import oracle


def test_starting_position_fills_the_array(start):
    '''The starting position lands on the array exactly where chess says it does.'''
    cb, board = start

    problems = oracle.grid_mismatches(board, cb)

    assert not problems, 'wrong piece codes: ' + '; '.join(
        f'{sq} expected {want!r} got {got!r}' for sq, want, got in problems
    )


def test_empty_squares_read_as_empty(start_board):
    '''An untouched square compares equal to ``EMPTY_SQUARE``.

    Every generator decides whether a square is free by comparing it against
    ``EMPTY_SQUARE`` - ``_slide_expand_move`` to stop a ray, ``gen_pawn_moves`` to
    allow a push.  Whatever the array holds for an empty square has to satisfy
    that comparison, or the generators silently treat the whole board as
    occupied and no piece moves more than one step.
    '''
    empty = [
        oracle.name((row, col))
        for row in range(2, 6)
        for col in range(8)
        if not oracle.is_empty(start_board, (row, col))
    ]

    assert not empty, (
        f'empty squares do not equal EMPTY_SQUARE=={game.EMPTY_SQUARE!r}: '
        f'{empty[:4]} hold {start_board.board[2, 0]!r}'
    )


def test_piece_codes_are_the_piece_signed_by_colour(start_board):
    '''A square holds the piece code, negated when the piece is white.

    Colour lives in the sign and the piece in the magnitude, which is what lets
    ``gen_pawn_moves`` recognise an enemy piece by multiplying the two signs
    together.  A square that drops the sign makes every piece look black.
    '''
    for square_, player, piece in (
        ('e1', game.WHITE, game.KING),
        ('d8', game.BLACK, game.QUEEN),
        ('a2', game.WHITE, game.PAWN),
        ('g8', game.BLACK, game.KNIGHT),
    ):
        found = oracle.code_at(start_board, oracle.at(square_))
        assert found == oracle.code(player, piece), (
            f'{square_} should hold {oracle.code(player, piece)}, holds {found}'
        )


def test_knight_and_bishop_are_not_interchangeable(start_board):
    '''Knights and bishops land on their own squares, not each other's.

    ``game.py`` numbers bishops 2 and knights 3; python-chess numbers them the
    other way round.  A translation slip would put bishops on b1/g1, and every
    move-generation test downstream would be asking the wrong piece to move.
    '''
    for corner in ('b1', 'g1', 'b8', 'g8'):
        code = oracle.code_at(start_board, oracle.at(corner))
        assert abs(code) == game.KNIGHT, f'{corner} should hold a knight, holds {code}'

    for corner in ('c1', 'f1', 'c8', 'f8'):
        code = oracle.code_at(start_board, oracle.at(corner))
        assert abs(code) == game.BISHOP, f'{corner} should hold a bishop, holds {code}'


def test_array_agrees_with_python_chess(fen):
    '''Across midgame positions, the array matches the real position square for square.'''
    cb, board = oracle.position(fen)

    problems = oracle.grid_mismatches(board, cb)

    assert not problems, f'{fen}: ' + '; '.join(
        f'{sq} expected {want!r} got {got!r}' for sq, want, got in problems[:8]
    )


def test_array_and_dictionaries_describe_one_position(fen):
    '''The array and the piece dictionaries are two views of the same board.

    ``Board`` keeps the position twice over, and the move methods update both.
    They have to start out agreeing, or a later divergence cannot be attributed.
    '''
    cb, board = oracle.position(fen)

    listed = {
        coord
        for pieces in (board.white_pieces, board.black_pieces)
        for coords in pieces.values()
        for coord in coords
    }
    occupied = {
        (row, col)
        for row in range(8)
        for col in range(8)
        if not oracle.is_empty(board, (row, col))
    }

    missing, extra = oracle.diff(listed, occupied)
    assert not missing and not extra, (
        f'{fen}: dictionaries and array disagree - '
        f'on the array only {missing}, in the dictionaries only {extra}'
    )
