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
PAWN_CHECKS_WHITE = ( (-1, -1), (-1, 1))

# Row step a pawn pushes by, and the row it is still allowed to push twice from
PAWN_STEP = {WHITE: -1, BLACK: 1}
PAWN_HOME_ROW = {WHITE: 6, BLACK: 1}

PLAYER = PAWN_STEP


PIECES = { "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING }
TENSOR_SIZE = 1_000

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
                    self.board[loc] = PLAYER[player] * piece

    def move_black_piece(self, piece, old_loc, new_loc, capture):
        self.board[old_loc] = EMPTY_SQUARE
        self.board[new_loc] = PLAYER[BLACK] * piece

        self.black_pieces[piece].remove(old_loc)
        self.black_pieces[piece].append(new_loc)

        # Checks for en-passant
        if piece == PAWN and old_loc[1] - new_loc[1] == 2:
            self.en_passant_white = new_loc
        else:
            self.en_passant_white = None

        # Performs Capture and w/ En-Passant
        if capture is True:
            if (piece == PAWN and self.board[new_loc] == EMPTY_SQUARE):
                capture_piece = self.board[self.en_passant_black]

                self.white_pieces[capture_piece].remove(self.en_passant_black)

            else:
                capture_piece = self.board[(new_loc)]
                self.white_pieces[capture_piece].remove(new_loc)


    def move_white_piece(self, piece, old_loc, new_loc, capture):
        self.board[old_loc] = EMPTY_SQUARE
        self.board[new_loc] = PLAYER[WHITE] * piece

        self.white_pieces[piece].remove(old_loc)
        self.white_pieces[piece].append(new_loc)

        if capture is True:
            self.black_pieces.remove(new_loc)

        # Checks for en-passant
        if piece == PAWN and new_loc[1] - old_loc[1] == 2:
            self.en_passant_black = new_loc
        else:
            self.en_passant_black = None

    def white_promotion(self, old_loc, new_loc, new_piece, capture):
        self.move_white_piece(PAWN, old_loc, new_loc, capture)

        # Update Memory
        self.board[new_loc] = PLAYER[WHITE] * new_piece
        self.white_pieces[PAWN].remove(new_loc)
        self.white_pieces[new_piece].append(new_loc)

    def black_promotion(self, old_loc, new_loc, new_piece, capture):
        self.move_black_piece(PAWN, old_loc, new_loc, capture)

        # Update Memory
        self.board[new_loc] = PLAYER[BLACK] * new_piece
        self.black_pieces[PAWN].remove(new_loc)
        self.black_pieces[new_pieces].append(new_loc)

    def _slide_expand_check(self, expanse, loc):
        moves = []
        x, y = loc

        for step in expanse:
            i = 1

            while (0 <= x + i * step[0] < 8 and 0 <= y + i * step[1] < 8):
                moves.append((x + i * step[0], y + i * step[1]))
                
                if self.board[(x + i * step[0], y + i * step[1])] != EMPTY_SQUARE:
                    break

                i += 1
        return moves

    def _slide_expand_move(self, expanse, loc, player):
        moves =[]
        x, y = loc

        for step in expanse:
            i = 1

            while (
                    0 <= x + i * step[0] < 8 and 
                    0 <= y + i * step[1] < 8 and
                    self.board[(x + i * step[0], y + i * step[1])] == EMPTY_SQUARE
                    ):
                moves.append((x + i * step[0], y + i * step[1]))
                i += 1
            
            # Add enemy capture
            if (0 <= x + i * step[0] < 8 and 
                0 <= y + i * step[1] < 8 and
                self.board[(x + i * step[0], y + i * step[1])] * PLAYER[player] < 0):
                moves.append((x + i * step[0], y + i * step[1]))

            return moves

    def gen_checks_rook(self, loc):
        return self._slide_expand_check(STRAIGHT_STEPS, loc)

    def gen_checks_bishop(self, loc):
        return self._slide_expand_check(DIAGONAL_STEPS, loc)

    def gen_checks_queen(self, loc):
        return self.gen_checks_rook(loc) + self.gen_checks_bishop(loc)

    def gen_moves_rook(self, loc, player):
        return self._slide_expand_move(STRAIGHT_STEPS, loc, player)

    def gen_moves_bishop(self, loc, player):
        return self._slide_expand_move(DIAGONAL_STEPS, loc, player)

    def gen_moves_knight(self, loc, player):
        x, y = loc
        return [
            (x + i_x, y + i_y)
            for i_x, i_y in KNIGHT_STEPS
            if (
                0 <= x + i_x < 8 and
                0 <= y + i_y < 8 and
                self.board[(x + i_x, y + i_y)] * PLAYER[player] <= 0
            )
        ]
    
    def gen_checks_knight(self, loc):
        x, y = loc
        return [
            (x + i_x, y + i_y)
            for i_x, i_y in KNIGHT_STEPS
            if (
                0 <= x + i_x < 8 and
                0 <= y + i_y < 8
            )
        ]

    def gen_moves_queen(self, loc, player):
        return self.gen_moves_rook(loc, player) + self.gen_moves_bishop(loc, player)

    def gen_pawn_moves(self, loc, player):
        x, y = loc
        moves = []

        step = PAWN_STEP[player]

        if self.board[(x + step, y)] == EMPTY_SQUARE:
            moves.append((x + step, y))

            if x == PAWN_HOME_ROW[player] and self.board[(x + 2 * step, y)] == EMPTY_SQUARE:
                moves.append((x + 2 * step, y ))

        # Capture
        for i_y in (-1, 1):
            if (
                0 <= y + i_y < 8 and
                self.board[(x + step, y + i_y)] * PLAYER[player] < 0
            ):
                moves.append((x + step, y + i_y))
        
        # En-Passant
        en_passant_loc = self.en_passant_black if player == WHITE else self.en_passant_white

        if en_passant_loc is not None and en_passant_loc[0] == x and abs(en_passant_loc[1] - y) == 1:
            moves.append(en_passant_loc[0], eno_passant_loc[1] + step)

        return moves


    def check_squares(self, player):
        pieces = self.black_pieces if player == BLACK else self.white_pieces

        checks = []
        for piece, locations in pieces.items():
            for loc in locations:
                if piece == ROOK: checks.extend(self.gen_checks_rook(loc))
                if piece == KNIGHT: checks.extend(self.gen_checks_knight(loc))
                if piece == BISHOP: checks.extend(self.gen_checks_bishop(loc))
                if piece == PAWN: checks.extend(self.get_checks_pawn(loc, player))
                if piece == KING: checks.extend(self.get_checks_king(loc))
                if piece == QUEEN: checks.extend(self.gen_checks_queen(loc))

        return checks

    def get_checks_pawn(self, loc, player):
        x, y = loc
        transform = PAWN_CHECKS_BLACK if player == BLACK else PAWN_CHECKS_WHITE
        return [
            (x + i_x, y + i_y) for i_x, i_y in transform
            if (
                0<= x + i_x < 8 and
                0<= y + i_y < 8
            )
        ]

    def get_checks_king(self, loc):
        x, y = loc
        return [
            (x + i_x, y + i_y) for i_x, i_y in KING_STEPS
            if (
                0 <= x + i_x < 8 and 
                0 <= y + i_y < 8
            )
        ]

    def gen_king_moves(self, loc, player):
        x, y = loc
        opp = WHITE if player == WHITE else BLACK
        check_squares = set(self.check_squares(opp))

        return [
            (x + i_x, y + i_y) for i_x, i_y in KING_STEPS
            if (
                 0 <= x + i_x < 8 and
                 0 <= y + i_y < 8 and
                 self.board[x + i_x, y + i_y] * PLAYER[player] <= 0 and
                 (x + i_x, y + i_y) not in check_squares
            )
        ]
