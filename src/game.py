'''A ``Board`` whose rules come from python-chess.

The representation is this project's own - a signed 8x8 ``int8`` array plus a
piece dictionary per side, addressed as ``(row, column)`` with row 0 == rank 8 -
and every mutation keeps that representation up to date.  Nothing here decides
what a legal move is: each generator translates the position into a
``chess.Board``, asks python-chess, and translates the answer back.

Three conventions have to be bridged, and they are all handled in one place:

Coordinates
    ``(row, column)`` numpy indices here, 0..63 from a1 upwards in python-chess.
    ``to_square`` and ``to_coord`` are the only two functions that know this.

Piece codes
    ``BISHOP == 2`` and ``KNIGHT == 3`` here, the other way round in
    python-chess, so codes always travel through ``CHESS_PIECE``.

En passant
    ``en_passant_white`` / ``en_passant_black`` are named for the side allowed
    to capture and hold the square the double-pushed pawn is *standing on*.
    python-chess stores the square it skipped over, for the side to move only.
'''

import chess
import numpy as np


WHITE = 0
BLACK = 1

EMPTY_SQUARE = 0
PAWN = 1
BISHOP = 2
KNIGHT = 3
ROOK = 4
QUEEN = 5
KING = 6


DIAGONAL_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
STRAIGHT_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
KING_STEPS = DIAGONAL_STEPS + STRAIGHT_STEPS
KNIGHT_STEPS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
PAWN_CHECKS_BLACK = ((1, -1), (1, 1))
PAWN_CHECKS_WHITE = ((-1, -1), (-1, 1))

# Row step a pawn pushes by, and the row it is still allowed to push twice from
PAWN_STEP = {WHITE: -1, BLACK: 1}
PAWN_HOME_ROW = {WHITE: 6, BLACK: 1}

#: The sign a square takes, by colour.  White is negative.
PLAYER = PAWN_STEP


PIECES = {"N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING}
TENSOR_SIZE = 1_000


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------

CHESS_PIECE = {
    PAWN: chess.PAWN,
    BISHOP: chess.BISHOP,
    KNIGHT: chess.KNIGHT,
    ROOK: chess.ROOK,
    QUEEN: chess.QUEEN,
    KING: chess.KING,
}
PIECE_FROM_CHESS = {value: key for key, value in CHESS_PIECE.items()}

CHESS_COLOR = {WHITE: chess.WHITE, BLACK: chess.BLACK}
COLOR_FROM_CHESS = {chess.WHITE: WHITE, chess.BLACK: BLACK}

#: Home row, king's column, and the two rook columns, by colour.
HOME_ROW = {WHITE: 7, BLACK: 0}
KING_COLUMN = 4
ROOK_COLUMNS = (0, 7)

#: Which slider a set of steps describes, for the two ``_slide_expand_*`` walkers.
SLIDER_FROM_STEPS = {
    frozenset(STRAIGHT_STEPS): ROOK,
    frozenset(DIAGONAL_STEPS): BISHOP,
    frozenset(KING_STEPS): QUEEN,
}


def to_square(loc):
    '''The python-chess square number of a ``(row, column)`` index.'''
    row, col = loc
    return chess.square(int(col), 7 - int(row))


def to_coord(square):
    '''The ``(row, column)`` index of a python-chess square number.'''
    return (7 - chess.square_rank(square), chess.square_file(square))


class Board:
    def __init__(self, black_pieces, white_pieces):
        self.black_pieces = black_pieces
        self.white_pieces = white_pieces

        self.en_passant_white = None        # Means that white can move in that dir if possible
        self.en_passant_black = None

        # Board Memory
        self.board = (np.ones((8, 8)) * EMPTY_SQUARE).astype(np.int8)

        # Fill Board
        for pieces, player in ((self.white_pieces, WHITE), (self.black_pieces, BLACK)):
            for piece, locations in pieces.items():
                for loc in locations:
                    self.board[tuple(loc)] = PLAYER[player] * piece

        # No castling field is handed to the constructor, so rights start out
        # read off the position and are narrowed from there as pieces move.
        self.castling_rights = {
            player: self._rights_from_placement(player) for player in (WHITE, BLACK)
        }

        self._cache = {}

    def _rights_from_placement(self, player):
        '''The rook columns ``player`` still looks entitled to castle with.

        The best a bare position can say: a king at home with a rook in the
        corner has probably not moved.  Once a game is under way ``_move``
        keeps this honest.
        '''
        if self.board[HOME_ROW[player], KING_COLUMN] != PLAYER[player] * KING:
            return set()

        return {
            col
            for col in ROOK_COLUMNS
            if self.board[HOME_ROW[player], col] == PLAYER[player] * ROOK
        }

    # ----------------------------------------------------------------------
    # The bridge to python-chess
    # ----------------------------------------------------------------------

    def _chess_board(self, player):
        '''This position as a ``chess.Board`` with ``player`` to move.

        Cached on the state it was built from, so a sweep over every piece in a
        position builds one board rather than one per piece.  Callers must treat
        the result as read-only.
        '''
        key = (
            player,
            self.board.tobytes(),
            self.en_passant_white,
            self.en_passant_black,
            frozenset((side, col) for side, cols in self.castling_rights.items() for col in cols),
        )

        if key not in self._cache:
            self._cache = {key: self._build(player)}

        return self._cache[key]

    def _build(self, player):
        cb = chess.Board(None)

        for row in range(8):
            for col in range(8):
                code = int(self.board[row, col])
                if code == EMPTY_SQUARE:
                    continue

                color = chess.BLACK if code > 0 else chess.WHITE
                cb.set_piece_at(
                    to_square((row, col)), chess.Piece(CHESS_PIECE[abs(code)], color)
                )

        cb.turn = CHESS_COLOR[player]
        cb.ep_square = self._ep_square(player)

        # clean_castling_rights() drops anything the placement contradicts, so
        # this only has to say which corners have not been given up.
        cb.castling_rights = chess.BB_EMPTY
        for side, columns in self.castling_rights.items():
            for col in columns:
                cb.castling_rights |= chess.BB_SQUARES[to_square((HOME_ROW[side], col))]
        cb.castling_rights = cb.clean_castling_rights()

        return cb

    def _ep_square(self, player):
        '''The square python-chess calls the en passant square, for ``player``.

        Our field holds the pawn's own square; python-chess wants the one it
        skipped over, which is a row further along the *capturing* side's
        direction of travel.
        '''
        pawn = self.en_passant_white if player == WHITE else self.en_passant_black

        if pawn is None:
            return None

        return to_square(pawn) + (8 if player == WHITE else -8)

    def _placed(self, loc, piece, player):
        '''This position, guaranteed to have ``player``'s ``piece`` on ``loc``.

        The generators are told which piece they are moving rather than reading
        it off the board, so a caller may ask what a rook would do from a square
        a rook is not standing on.
        '''
        cb = self._chess_board(player)
        square = to_square(loc)
        want = chess.Piece(CHESS_PIECE[piece], CHESS_COLOR[player])

        if cb.piece_at(square) == want:
            return cb

        cb = cb.copy(stack=False)
        cb.set_piece_at(square, want)
        return cb

    def _player_at(self, loc):
        '''Whose piece stands on a square, defaulting to white when it is empty.'''
        code = int(self.board[tuple(loc)])
        return BLACK if code > 0 else WHITE

    def _targets(self, loc, piece, player, legal=False):
        '''Where ``player``'s ``piece`` on ``loc`` may go, as coordinate pairs.

        Promotions collapse onto their target square, since these generators
        deal in squares rather than (square, promotion piece) pairs.
        '''
        cb = self._placed(loc, piece, player)
        from_mask = chess.BB_SQUARES[to_square(loc)]

        moves = (
            cb.generate_legal_moves(from_mask=from_mask)
            if legal
            else cb.generate_pseudo_legal_moves(from_mask=from_mask)
        )

        seen = {}
        for move in moves:
            seen[to_coord(move.to_square)] = None

        return list(seen)

    def _attacks(self, loc, piece, player):
        '''Every square ``player``'s ``piece`` on ``loc`` attacks, occupied or not.'''
        cb = self._placed(loc, piece, player)
        return [to_coord(square) for square in cb.attacks(to_square(loc))]

    # ----------------------------------------------------------------------
    # Applying moves
    # ----------------------------------------------------------------------

    def move_black_piece(self, piece, old_loc, new_loc, capture):
        self._move(BLACK, piece, old_loc, new_loc, capture)

    def move_white_piece(self, piece, old_loc, new_loc, capture):
        self._move(WHITE, piece, old_loc, new_loc, capture)

    def _move(self, player, piece, old_loc, new_loc, capture):
        old_loc, new_loc = tuple(old_loc), tuple(new_loc)
        mine = self.white_pieces if player == WHITE else self.black_pieces
        theirs = self.black_pieces if player == WHITE else self.white_pieces

        if capture:
            # A pawn capturing onto an empty square is taking en passant, and the
            # pawn it takes is beside where it started, not where it lands.
            taken = new_loc
            if (
                piece == PAWN
                and new_loc[1] != old_loc[1]
                and self.board[new_loc] == EMPTY_SQUARE
            ):
                taken = (old_loc[0], new_loc[1])

            self._capture(theirs, taken)

        self.board[old_loc] = EMPTY_SQUARE
        self.board[new_loc] = PLAYER[player] * piece

        mine[piece].remove(old_loc)
        mine[piece].append(new_loc)

        # Castling is a king move of two files, and the rook follows it
        if piece == KING and abs(new_loc[1] - old_loc[1]) == 2:
            self._move_castling_rook(mine, player, old_loc, new_loc)

        self._revoke_castling(player, piece, old_loc)

        # The right to capture en passant lasts exactly one ply: whoever just
        # moved has lost theirs, and a double push hands one to the opponent.
        double_push = piece == PAWN and abs(new_loc[0] - old_loc[0]) == 2

        if player == WHITE:
            self.en_passant_white = None
            self.en_passant_black = new_loc if double_push else None
        else:
            self.en_passant_black = None
            self.en_passant_white = new_loc if double_push else None

    def _capture(self, pieces, loc):
        '''Take the piece on ``loc`` off the array and out of its dictionary.'''
        code = int(self.board[loc])

        if code == EMPTY_SQUARE:
            return

        self.board[loc] = EMPTY_SQUARE
        pieces[abs(code)].remove(loc)

        # A rook taken in its corner takes that side's castling right with it
        if abs(code) == ROOK:
            self._revoke_castling(BLACK if code > 0 else WHITE, ROOK, loc)

    def _revoke_castling(self, player, piece, old_loc):
        '''Give up whatever castling right leaving ``old_loc`` costs ``player``.

        A king gives up both; a rook gives up its own side only.  Nothing here
        checks that the piece had ever been home - a rook that was never on its
        corner discards a right that was not in the set to begin with.
        '''
        if piece == KING:
            self.castling_rights[player] = set()
        elif piece == ROOK and old_loc[0] == HOME_ROW[player]:
            self.castling_rights[player].discard(old_loc[1])

    def _move_castling_rook(self, mine, player, old_loc, new_loc):
        row = old_loc[0]
        corner, lands_on = ((row, 7), (row, 5)) if new_loc[1] > old_loc[1] else ((row, 0), (row, 3))

        self.board[corner] = EMPTY_SQUARE
        self.board[lands_on] = PLAYER[player] * ROOK

        mine[ROOK].remove(corner)
        mine[ROOK].append(lands_on)

    def white_promotion(self, old_loc, new_loc, new_piece, capture):
        self._promote(WHITE, old_loc, new_loc, new_piece, capture)

    def black_promotion(self, old_loc, new_loc, new_piece, capture):
        self._promote(BLACK, old_loc, new_loc, new_piece, capture)

    def _promote(self, player, old_loc, new_loc, new_piece, capture):
        self._move(player, PAWN, old_loc, new_loc, capture)

        new_loc = tuple(new_loc)
        mine = self.white_pieces if player == WHITE else self.black_pieces

        # Update Memory
        self.board[new_loc] = PLAYER[player] * new_piece
        mine[PAWN].remove(new_loc)
        mine[new_piece].append(new_loc)

    # ----------------------------------------------------------------------
    # Move generation - pseudo-legal, except the king
    # ----------------------------------------------------------------------

    def _slide_expand_move(self, expanse, loc, player):
        return self._targets(loc, SLIDER_FROM_STEPS[frozenset(expanse)], player)

    def gen_moves_rook(self, loc, player):
        return self._targets(loc, ROOK, player)

    def gen_moves_bishop(self, loc, player):
        return self._targets(loc, BISHOP, player)

    def gen_moves_queen(self, loc, player):
        return self._targets(loc, QUEEN, player)

    def gen_moves_knight(self, loc, player):
        return self._targets(loc, KNIGHT, player)

    def gen_pawn_moves(self, loc, player):
        return self._targets(loc, PAWN, player)

    def gen_king_moves(self, loc, player):
        '''The king's *legal* moves - it is the one piece that may not ignore check.'''
        return self._targets(loc, KING, player, legal=True)

    # ----------------------------------------------------------------------
    # The attack map - where a side covers, whoever is standing there
    # ----------------------------------------------------------------------

    def _slide_expand_check(self, expanse, loc):
        piece = SLIDER_FROM_STEPS[frozenset(expanse)]
        return self._attacks(loc, piece, self._player_at(loc))

    def gen_checks_rook(self, loc):
        return self._attacks(loc, ROOK, self._player_at(loc))

    def gen_checks_bishop(self, loc):
        return self._attacks(loc, BISHOP, self._player_at(loc))

    def gen_checks_queen(self, loc):
        return self._attacks(loc, QUEEN, self._player_at(loc))

    def gen_checks_knight(self, loc):
        return self._attacks(loc, KNIGHT, self._player_at(loc))

    def get_checks_pawn(self, loc, player):
        return self._attacks(loc, PAWN, player)

    def get_checks_king(self, loc):
        return self._attacks(loc, KING, self._player_at(loc))

    def check_squares(self, player):
        '''Every square ``player`` attacks, counted once per attacker.

        Read off the array rather than the piece dictionaries, so that a
        dictionary that has drifted out of step shows up as a dictionary
        problem rather than as a king wandering into check.
        '''
        cb = self._chess_board(player)

        checks = []
        for square in chess.scan_forward(cb.occupied_co[CHESS_COLOR[player]]):
            checks.extend(to_coord(attacked) for attacked in cb.attacks(square))

        return checks
