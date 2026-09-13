import chess
import chess.svg

PIECE_MAP = {
    1: chess.PAWN,
    2: chess.KNIGHT,
    3: chess.BISHOP,
    4: chess.ROOK,
    5: chess.QUEEN,
    6: chess.KING,
}

def fog_chess_board(board, color):
    board = board.get_visible_squares(color)
    return array_to_board(board)

def chess_board(board):
    return array_to_board(board.get_squares())

def array_to_board(arr):
    board = chess.Board(None)  # empty board

    for row in range(8):
        for col in range(8):
            value = arr[row][col]

            if value == 0:
                continue

            color = chess.WHITE if value > 0 else chess.BLACK
            piece_type = PIECE_MAP[abs(value)]

            # If row 0 in your array represents rank 8:
            rank = 7 - row
            file = col

            square = chess.square(file, rank)

            board.set_piece_at(
                square,
                chess.Piece(piece_type, color)
            )

    return board
