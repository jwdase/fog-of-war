'''Helpers for reading and comparing the two board-state representations.

The board state in ``src/data/data.py`` is held twice over:

* ``ChessState.data`` - an 8x8 integer array of piece codes, indexed
  ``[row, column]`` with row 0 == rank 8 and column 0 == file 'a'.
* ``ChessState.white_pieces`` / ``.black_pieces`` - ``{piece_code: [square, ...]}``
  dictionaries holding the same information in algebraic notation.

Tests check each representation on its own terms, then check the two agree.
'''

from src.data.data import COORDINATES, EMPTY_SQUARE

FILES = "abcdefgh"
RANKS = "12345678"

#: Every square of the board, in algebraic notation.
ALL_SQUARES = [f + r for f in FILES for r in RANKS]


def square(row, col):
    '''Decode an array index back to algebraic notation.

    Mirrors the expression the move generators use to build their return values.
    '''
    return chr(col + ord('a')) + str(8 - row)


def piece_at(data, coord):
    '''Piece code standing on ``coord`` in a board array.'''
    return data[COORDINATES[coord]]


def occupied_squares(data):
    '''Set of algebraic squares holding something other than an empty square.'''
    return {
        square(row, col)
        for row in range(8)
        for col in range(8)
        if data[row, col] != EMPTY_SQUARE
    }


def all_listed_squares(pieces):
    '''Flatten a ``{piece: [square, ...]}`` dictionary into a list of squares.'''
    return [coord for squares in pieces.values() for coord in squares]
