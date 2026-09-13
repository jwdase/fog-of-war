'''python-chess as the ground truth for what a legal chess move is.

Everything here exists to answer one question: *given a position, what does
real chess allow?*  ``src/game.py`` is then asked the same question and the two
answers are compared.  No expected move list in this suite is written by hand -
they all come out of ``python-chess`` - so the tests cannot inherit a
misunderstanding of the rules from whoever wrote them.

Three conventions have to be bridged.

Coordinates
    ``src/game.py`` (like ``src/data.py``) addresses squares as ``(row, column)``
    numpy indices, row 0 == rank 8 and column 0 == file 'a'.  That is what
    ``PAWN_STEP = {WHITE: -1}`` and ``PAWN_HOME_ROW = {WHITE: 6, BLACK: 1}``
    describe: white pawns start on row 6 and advance towards row 0.
    ``python-chess`` numbers squares 0..63 from a1 upwards instead.

Piece codes
    The two modules disagree about knights and bishops - ``game.BISHOP == 2``
    and ``game.KNIGHT == 3``, where ``chess.KNIGHT == 2`` and
    ``chess.BISHOP == 3`` - so piece codes are always translated through
    ``PIECE_FROM_CHESS``, never passed straight across.

Square encoding
    ``game.Board.board`` is an ``int8`` array holding one signed number per
    square: the sign is the colour (white negative, black positive), the
    magnitude is the piece code, and ``0`` is an empty square.  ``COLOR_SIGN``
    and ``code`` state that here rather than importing ``game.PLAYER``, so that
    a change to how ``game.py`` signs a square fails a test instead of being
    followed silently.

Move semantics
    Different questions get different oracles, because the functions under test
    are answering different questions:

    * ``pseudo_legal_targets`` - where may this piece go, ignoring whether the
      move leaves the mover's own king hanging.  The per-piece generators
      (``gen_moves_rook``, ``gen_pawn_moves``, ...) do no self-check filtering,
      so this is what they are measured against.
    * ``legal_targets`` - the same, restricted to moves that are actually legal.
      ``gen_king_moves`` claims this stronger property, since it filters out
      squares the opponent attacks, so it is measured against this.
    * ``attacked_coords`` - every square a side attacks, occupied or not.  This
      is what ``check_squares`` is for, and it is deliberately *not* a move
      list: a rook defending its own neighbour attacks that square without
      being able to move there.
'''

import random

import chess
import numpy as np

from src import game

FILES = 'abcdefgh'

#: game.py's piece codes, keyed by python-chess's.  Knight and bishop differ.
PIECE_FROM_CHESS = {
    chess.PAWN: game.PAWN,
    chess.BISHOP: game.BISHOP,
    chess.KNIGHT: game.KNIGHT,
    chess.ROOK: game.ROOK,
    chess.QUEEN: game.QUEEN,
    chess.KING: game.KING,
}

#: Every piece code game.py knows, for building complete piece dictionaries.
ALL_PIECES = tuple(PIECE_FROM_CHESS.values())

COLOR_FROM_CHESS = {chess.WHITE: game.WHITE, chess.BLACK: game.BLACK}
COLOR_TO_CHESS = {game.WHITE: chess.WHITE, game.BLACK: chess.BLACK}

#: The sign game.py gives a square, by colour.  White is negative.
COLOR_SIGN = {game.WHITE: -1, game.BLACK: 1}


def code(player, piece):
    '''The number ``game.Board.board`` should hold for one of ``player``'s pieces.'''
    return COLOR_SIGN[player] * piece


def code_at(board, coord_):
    '''The number on one square of a ``game.Board``, as a plain int.'''
    return int(board.board[tuple(coord_)])


def is_empty(board, coord_):
    '''Whether a square of a ``game.Board`` reads as empty.'''
    return code_at(board, coord_) == game.EMPTY_SQUARE


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def coord(square):
    '''The ``(row, column)`` index of a python-chess square number.'''
    return (7 - chess.square_rank(square), chess.square_file(square))


def square(coord_):
    '''The python-chess square number of a ``(row, column)`` index.'''
    row, col = coord_
    return chess.square(col, 7 - row)


def name(coord_):
    '''Algebraic name of a ``(row, column)`` index, so failures stay readable.'''
    row, col = coord_
    if not (0 <= row < 8 and 0 <= col < 8):
        return f'off-board{tuple(coord_)}'
    return f'{FILES[col]}{8 - row}'


def names(coords):
    '''A collection of coordinates as sorted algebraic names.'''
    return sorted(name(c) for c in coords)


def at(name_):
    '''The ``(row, column)`` index of an algebraic square name.'''
    return coord(chess.parse_square(name_))


def at_all(*names_):
    '''A set of ``(row, column)`` indices for some algebraic square names.'''
    return {at(n) for n in names_}


# --------------------------------------------------------------------------
# Building the subject under test
# --------------------------------------------------------------------------

def piece_dicts(cb):
    '''The ``{piece_code: [coord, ...]}`` pair a ``game.Board`` is built from.

    Every piece code is present as a key even for a side that has none left, so
    that ``check_squares`` can iterate and ``move_*_piece`` can ``remove()``
    without tripping over a missing key.
    '''
    built = {
        chess.WHITE: {piece: [] for piece in ALL_PIECES},
        chess.BLACK: {piece: [] for piece in ALL_PIECES},
    }

    for sq, piece in cb.piece_map().items():
        built[piece.color][PIECE_FROM_CHESS[piece.piece_type]].append(coord(sq))

    return built[chess.WHITE], built[chess.BLACK]


def board_from(cb):
    '''A ``game.Board`` holding the position a ``chess.Board`` holds.

    Two things are easy to get backwards here.  ``game.Board`` takes *black*
    pieces first, and it records en passant as "the square the pawn that just
    double-pushed is standing on", named for the side allowed to capture it -
    where python-chess records the square that pawn skipped over.
    '''
    white, black = piece_dicts(cb)
    board = game.Board(black, white)

    if cb.ep_square is not None:
        if cb.turn == chess.WHITE:
            board.en_passant_white = coord(cb.ep_square - 8)
        else:
            board.en_passant_black = coord(cb.ep_square + 8)

    return board


def position(fen):
    '''An ``(oracle, subject)`` pair for a FEN: a chess.Board and a game.Board.'''
    cb = chess.Board(fen)
    return cb, board_from(cb)


def to_move(cb, player):
    '''The same position with ``player`` (a game.py colour) to move.

    python-chess only generates moves for the side to move, so asking about a
    black knight in a white-to-move position needs the turn flipped.  An en
    passant right belongs to the side to move, so it is dropped when the turn
    changes - meaning en passant is only ever tested in positions already set
    up with the capturing side to move.
    '''
    want = COLOR_TO_CHESS[player]
    if cb.turn == want:
        return cb

    flipped = cb.copy()
    flipped.turn = want
    flipped.ep_square = None
    return flipped


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------

def pseudo_legal_targets(cb, coord_, *, castling=True):
    '''Squares the piece on ``coord_`` may move to, ignoring self-check.

    Promotions collapse onto their target square, because the generators under
    test return squares rather than (square, promotion piece) pairs.
    '''
    moves = cb.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[square(coord_)])
    return {coord(m.to_square) for m in moves if castling or not cb.is_castling(m)}


def legal_targets(cb, coord_, *, castling=True):
    '''Squares the piece on ``coord_`` may *legally* move to.'''
    frm = square(coord_)
    return {
        coord(m.to_square)
        for m in cb.legal_moves
        if m.from_square == frm and (castling or not cb.is_castling(m))
    }


def attacked_coords(cb, player):
    '''Every square ``player`` attacks, whether or not a piece stands on it.'''
    color = COLOR_TO_CHESS[player]
    return {coord(sq) for sq in chess.SQUARES if cb.is_attacked_by(color, sq)}


def squares_of(cb, player, piece=None):
    '''Coordinates where ``player`` has a piece, optionally of one kind only.'''
    color = COLOR_TO_CHESS[player]
    return [
        coord(sq)
        for sq, found in cb.piece_map().items()
        if found.color == color
        and (piece is None or PIECE_FROM_CHESS[found.piece_type] == piece)
    ]


# --------------------------------------------------------------------------
# Applying moves, and checking the result against the oracle
# --------------------------------------------------------------------------

def apply(board, cb, move):
    '''Play ``move`` on a ``game.Board``, driving it the way it expects.

    Chooses between the white/black and plain/promotion entry points, and works
    out the ``capture`` flag from the position rather than from the caller, so a
    test cannot accidentally lie to the board about what it is doing.
    '''
    moved = cb.piece_at(move.from_square)
    player = COLOR_FROM_CHESS[moved.color]
    capture = cb.is_capture(move)
    frm, to = coord(move.from_square), coord(move.to_square)

    if move.promotion:
        promote_to = PIECE_FROM_CHESS[move.promotion]
        if player == game.WHITE:
            board.white_promotion(frm, to, promote_to, capture)
        else:
            board.black_promotion(frm, to, promote_to, capture)
    elif player == game.WHITE:
        board.move_white_piece(PIECE_FROM_CHESS[moved.piece_type], frm, to, capture)
    else:
        board.move_black_piece(PIECE_FROM_CHESS[moved.piece_type], frm, to, capture)


def expected_codes(cb):
    '''The 8x8 grid of signed piece codes a position should produce.'''
    grid = np.full((8, 8), game.EMPTY_SQUARE, dtype=np.int8)

    for sq, piece in cb.piece_map().items():
        grid[coord(sq)] = code(
            COLOR_FROM_CHESS[piece.color], PIECE_FROM_CHESS[piece.piece_type]
        )

    return grid


def grid_mismatches(board, cb):
    '''Squares where a ``game.Board``'s array disagrees with the real position.

    Returns ``[(square name, expected code, actual code), ...]``.
    '''
    expected = expected_codes(cb)
    return [
        (name((row, col)), int(expected[row, col]), code_at(board, (row, col)))
        for row in range(8)
        for col in range(8)
        if code_at(board, (row, col)) != expected[row, col]
    ]


def dict_mismatches(board, cb):
    '''Piece codes whose listed squares disagree with the real position.

    Returns ``[(colour, piece code, expected names, actual names), ...]``.
    '''
    white, black = piece_dicts(cb)
    problems = []

    for label, expected, actual in (
        ('white', white, board.white_pieces),
        ('black', black, board.black_pieces),
    ):
        for piece in ALL_PIECES:
            want = set(expected[piece])
            got = set(actual.get(piece, []))
            if want != got:
                problems.append((label, piece, names(want), names(got)))

    return problems


# --------------------------------------------------------------------------
# Position corpus
# --------------------------------------------------------------------------

def random_positions(count=20, *, seed=7, max_ply=70, min_ply=6):
    '''``(label, fen)`` pairs drawn from random legal play.

    A generator that only survives the starting position is not much of a move
    generator, so the sweep tests run over real midgame positions - captures on
    offer, pieces pinned, pawns one step from promoting - rather than over
    positions chosen by whoever wrote the test.  Fixed seed, so a failure is
    reproducible.
    '''
    rng = random.Random(seed)
    cb = chess.Board()
    found = []

    while len(found) < count:
        if cb.is_game_over(claim_draw=False) or cb.ply() >= max_ply:
            cb = chess.Board()

        cb.push(rng.choice(list(cb.legal_moves)))

        if cb.ply() >= min_ply:
            found.append((f'seed{seed}-ply{cb.ply()}', cb.fen()))

    return found


def diff(actual, expected):
    '''``(missing, unexpected)`` square names between a generator and the oracle.

    Compared as sets: a generator is free to hand back the same square twice
    (``gen_moves_queen`` concatenates two lists, ``check_squares`` accumulates
    across pieces) and that is not what these tests are about.
    '''
    produced = {tuple(c) for c in actual}
    wanted = {tuple(c) for c in expected}
    return names(wanted - produced), names(produced - wanted)


def attacks_from(cb, coord_):
    '''Squares the single piece on ``coord_`` attacks.

    The right oracle for ``get_checks_pawn`` and ``get_checks_king``: an attack
    map, so a square counts whether it is empty, holds an enemy piece, or holds
    one of your own.  A pawn attacks the two diagonals in front of it even with
    nothing to capture, and a king attacks all eight neighbours even when its
    own rook is standing on one of them.
    '''
    return {coord(sq) for sq in cb.attacks(square(coord_))}


def expected_ep(cb):
    '''``(en_passant_white, en_passant_black)`` as a ``game.Board`` should hold them.

    game.py names each field for the side allowed to capture, and stores the
    square the double-pushed pawn is standing on.  python-chess stores the square
    it skipped over, and only ever for the side to move - so the side to move is
    what decides which of the two fields should be set.
    '''
    if cb.ep_square is None:
        return None, None

    if cb.turn == chess.WHITE:
        return coord(cb.ep_square - 8), None

    return None, coord(cb.ep_square + 8)
