import torch
import numpy as np
import pandas as pd

from main import DATA_DIR

files = sorted(DATA_DIR.glob("*.parquet"))
output_dir = DATA_DIR / "processed"

output_dir.mkdir(parents=True, exist_ok=True)


'''
Coordinates

A8
        
        H1
'''

WHITE = 0
BLACK = 1


EMPTY_SQUARE = 0
PAWN = 1
BISHOP = 2
KNIGHT = 3
ROOK = 4
QUEEN = 5
KING = 6

COORDINATES = {
    char_ + num_ : (i_h, i_w)
    for i_h, num_ in enumerate("87654321")
    for i_w, char_ in enumerate("abcdefgh")
}

def pawn_moves(coord, ):
    '''Returns a list of all possible pawn moves from a given coordinate'''
    x, y = coord
    moves = []
    if x > 0:
        moves.append((x - 1, y))  # Move forward
        if x == 6:  # Initial double move
            moves.append((x - 2, y))
    return [(chr(y_ + ord('a')) + str(8 - x_)) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]

def knight_moves(coord):
    '''Returns a list of all possible knight moves from a given coordinate'''
    x, y = coord
    moves = [
        (x + 2, y + 1), (x + 2, y - 1),
        (x - 2, y + 1), (x - 2, y - 1),
        (x + 1, y + 2), (x + 1, y - 2),
        (x - 1, y + 2), (x - 1, y - 2)
    ]
    return [(chr(y_ + ord('a')) + str(8 - x_)) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]

def bishop_moves(coord):
    '''Returns a list of all possible bishop moves from a given coordinate'''
    x, y = coord
    moves = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for step in range(1, 8):
                new_x, new_y = x + dx * step, y + dy * step
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    moves.append((new_x, new_y))
                else:
                    break
    return [(chr(y_ + ord('a')) + str(8 - x_)) for x_, y_ in moves]

def rook_moves(coord):
    '''Returns a list of all possible rook moves from a given coordinate'''
    x, y = coord
    moves = []
    for dx in [-1, 1]:
        for step in range(1, 8):
            new_x = x + dx * step
            if 0 <= new_x < 8:
                moves.append((new_x, y))
            else:
                break
    for dy in [-1, 1]:
        for step in range(1, 8):
            new_y = y + dy * step
            if 0 <= new_y < 8:
                moves.append((x, new_y))
            else:
                break
    return [(chr(y_ + ord('a')) + str(8 - x_)) for x_, y_ in moves]

def queen_moves(coord):
    '''Returns a list of all possible queen moves from a given coordinate'''
    return bishop_moves(coord) + rook_moves(coord)

def king_moves(coord):
    '''Returns a list of all possible king moves from a given coordinate'''
    x, y = COORDINATES[coord]
    moves = [
        (x + 1, y), (x - 1, y),
        (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x + 1, y - 1),
        (x - 1, y + 1), (x - 1, y - 1)
    ]
    return [(chr(y_ + ord('a')) + str(8 - x_)) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]


MOVE_FUNCTIONS = {
    PAWN: pawn_moves,
    KNIGHT: knight_moves,
    BISHOP: bishop_moves,
    ROOK: rook_moves,
    QUEEN: queen_moves,
    KING : king_moves,
}

class ChessState:
    def __init__(self):
        black_pieces, white_pieces = self.gen_game()

        self.black_pieces = black_pieces
        self.white_pieces = white_pieces

    def write_board(self):
        '''Lists out a board state'''
        data = np.zeros((8, 8), dtype=int)

        for piece, locations in self.white_pieces.items():
            for location in locations:
                data[x, y] = piece

        for piece, locations in self.black_pieces.items():
            for location in locations:
                data[x, y] = piece

        return data

    def gen_game(self):
        white_pieces = {
            PAWN : [(6, i) for i in range(0, 8)],
            ROOK : [(7, 0), (7, 7)],
            KNIGHT : [(7, 1), (7, 6)],
            BISHOP : [(7, 2), (7, 5)],
            QUEEN : [(7, 3)],
            KING : [7, 4],
        }

        black_pieces = {
            PAWN : [(1, i) for i in range(0, 8)],
            ROOK : [(0, 0), (0, 7)],
            KNIGHT : [(0, 1), (0, 6)],
            BISHOP : [(0, 2), (0, 5)],
            QUEEN : [(0, 3)],
            KING : [(0, 4)],
        }

        return black_pieces, white_pieces

    def move_piece(self, player_pieces, opponent_pieces, piece, column, new_location, capture):
        '''
        Resolves original location of a given piece

        Quick Review of the Values
        -> player_pieces: Dictionary of the player's pieces and their locations (a8)
        -> new_coordinates: The target coordinates to move the piece to (a8)

        NOTE: location keys -> (ab) chess style layout
        '''

        if piece not in pieces:
            raise ValueError(f"Piece {piece} not found in pieces.")
        
        # Get the move function for the piece and the target coordinates
        move_function = MOVE_FUNCTIONS[piece]
        new_coord = COORDINATES[new_location]

        # Find the original location
        prev_location = None
        for chess_move in pieces[piece]:

            # Eliminate locations that don't match the specified column
            if column is not None and chess_move[0] != column:
                continue

            # Then coordinate is valid, ensure that not multiple copies
            if new_coord in move_function(location):
                if prev_location is not None:
                    raise ValueError(f"Multiple pieces of type {piece} can move to {new_coordinates}.")

                prev_location = location
        
        # Ensure we have found a valid original location
        if prev_location is None:
            raise ValueError(f"No piece of type {piece} can move to {new_coordinates}.")

        # Update the pieces dictionary
        if capture is True:

            # Remove the captured piece from the opponent's pieces
            for opp_piece, opp_locations in opponent_pieces.items():
                if new_coord in opp_locations:
                    opp_locations.remove(new_coord)
                    break
            else:
                raise ValueError(f"No opponent piece found at {new_coordinates} to capture.")

        # Update the player's pieces
        player_pieces[piece].remove(old_location)
        player_pieces[piece].append(new_location)


    def move_piece_white(self, piece, new_coordinates, capture, column):
        self.move_piece(self.white_pieces, self.black_pieces, piece, column, new_coordinates, capture)

    def move_piece_black(self, piece, new_coordinates, capture, column):
        self.move_piece(self.black_pieces, self.white_pieces, piece, column, new_coordinates, capture)

    def gen_state(self):
        game_state_white = self.visible_positions(self.white_pieces)
        game_state_black = self.visible_positions(self.black_pieces)

        return game_state_white, game_state_black, self.data


    def visible_positions(self, pieces):
        positions = np.zeros((8, 8), dtype=int)

        opponent_pieces = {
                cord : piece 
                for cord in coordinates
                for piece, coordinates in pieces.items()
            }

        for piece, locations in pieces.items():

            # Uncover our Current Pieces
            for location in locations:
                positions[location] = piece
                
                # Travel to all visable positions for piece
                for move in MOVE_FUNCTIONS[piece](location):
                    positions[move] = opponent_pieces.get(move, EMPTY_SQUARE)

        return positions



MOVER = {WHITE: ChessState.move_piece_white, BLACK: ChessState.move_piece_black}
PIECES = { "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING }
TENSOR_SIZE = 1_000


class ChessGame:
    def __init__(self, files):
        self.files = iter(files)

        try:
            self.current_file = next(self.files)
        except StopIteration:
            raise ValueError("No parquet files found in the data directory.")

        self.games = self.load_data()

        self.input_data = []
        self.output_data = []


    def load_data(self):
        '''Yield a list of moves for each game'''
        df = pd.read_parquet(self.current_file)

        for moves in df["moves"]:

            opt = []
            for move in moves.split(" "):
                try:
                    float(move)
                except ValueError:
                    opt.append(move)

            yield opt

    def run_game(self, game):
        game_data_white = []
        game_data_black = []
        game_board_layout = []

        for i, move in enumerate(game):
            player = WHITE if i % 2 == 0 else BLACK
            self.make_move(move, player)

            game_state_white, game_state_black, board_state = self.state.gen_state()

            game_data_white.append(game_state_white)
            game_data_black.append(game_state_black)
            game_board_layout.append(board_state)
        
        # Add Game White POV
        self.input_data.append(np.array(game_data_white, dtype=np.int8))
        self.input_data.append(np.array(game_data_black, dtype=np.int8))
        
        # Add Game Black POV
        self.output_data.append(np.array(game_board_layout, dtype=np.int8))
        self.output_data.append(np.array(game_board_layout, dtype=np.int8))


    def run_data_process(self):
        while True:
            try:
                game = next(self.games)
            except StopIteration:
                try:
                    self.current_file = next(self.files)
                    self.games = self.load_data()
                    game = next(self.games)
                except StopIteration:
                    break

            self.state = ChessState()
            self.run_game(game)

            if len(self.input_data) >= TENSOR_SIZE:
                self.save_data()


    def save_data(self):

        torch.save(
            {
                "input_data" : torch.tensor(self.input_data[0:TENSOR_SIZE], dtype=torch.int8),
                "output_data" : torch.tensor(self.output_data[0:TENSOR_SIZE], dtype=torch.int8),
            }, output_dir / f"{self.current_file.stem}.pt")

        self.input_data = self.input_data[TENSOR_SIZE:]
        self.output_data = self.output_data[TENSOR_SIZE:]

    def make_move(self, move, player):
        '''Make a move for the given player (white or black)'''

        assert player in {WHITE, BLACK}, "Player must be either WHITE or BLACK"

        if move == "0-0":
            
            if player == WHITE:
                self.state.move_piece_white(KING, "g1", capture=False, column=None)
                self.state.move_piece_white(ROOK, "f1", capture=False, column=None)
            else:
                self.state.move_piece_black(KING, "g8", capture=False, column=None)
                self.state.move_piece_black(ROOK, "f8", capture=False, column=None)
            return

        if move == "0-0-0":
            if player == BLACK:
                self.state.move_piece_white(KING, "c1", capture=False, column=None)
                self.state.move_piece_white(ROOK, "d1", capture=False, column=None)
            else:
                self.state.move_piece_black(KING, "c8", capture=False, column=None)
                self.state.move_piece_black(ROOK, "d8", capture=False, column=None)
            return
        
        # Remove Check or Checkmate
        move = move.replace("+", "").replace("#", "")
        
        # Finds the capture
        if 'x' in move:
            capture = True
            move = move.replace('x', '')
        
        
        # Only length 4 will have a column
        if len(move) == 4:
            column = move[1]
            move = move[0] + move[2:]
        
        # Get the piece
        if move[0].islower():
            piece = PAWN
        else:
            piece = PIECES[move[0]]
            move = move[1:]

        assert len(move) == 2, "Move must be in the format 'e4' or 'Nf3'"
        
        # Run move function
        MOVER[player](self.state, piece, move, capture=capture, column=column)
        return  




        

if __name__ == "__main__":
    chessgame = ChessGame(files)
