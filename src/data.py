

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
                data[location] = piece

        for piece, locations in self.black_pieces.items():
            for location in locations:
                data[location] = piece

        return data

    def gen_game(self):
        white_pieces = {
            PAWN : [(6, i) for i in range(0, 8)],
            ROOK : [(7, 0), (7, 7)],
            KNIGHT : [(7, 1), (7, 6)],
            BISHOP : [(7, 2), (7, 5)],
            QUEEN : [(7, 3),],
            KING : [(7, 4),],
        }

        black_pieces = {
            PAWN : [(1, i) for i in range(0, 8)],
            ROOK : [(0, 0), (0, 7)],
            KNIGHT : [(0, 1), (0, 6)],
            BISHOP : [(0, 2), (0, 5)],
            QUEEN : [(0, 3),],
            KING : [(0, 4),],
        }

        return black_pieces, white_pieces

    def move_piece(self, player, piece, column, new_location, capture, promotion):
        '''
        Resolves original location of a given piece

        Quick Review of the Values
        -> player_pieces: Dictionary of the player's pieces and their locations (a8)
        -> new_coordinates: The target coordinates to move the piece to (a8)

        NOTE: location keys -> (ab) chess style layout
        '''

        assert player in {BLACK, WHITE}, f"Player must be black or white no {player}"

        if player == WHITE:
            player_pieces = self.white_pieces
            opponent_pieces = self.black_pieces
        else:
            player_pieces = self.black_pieces
            opponent_pieces = self.white_pieces

        if piece not in player_pieces:
            raise ValueError(f"Piece {piece} not found in pieces.")
        
        # Get the move function for the piece and the target coordinates
        new_coord = COORDINATES[new_location]
        occupied_squares = self.occupancy()

        # Find the original location
        prev_location = None
        for coord in player_pieces[piece]:

            # Eliminate locations that don't match the specified column
            if column is not None and coord[0] != COLUMNS[column]:
                continue

            # Then coordinate is valid, ensure that not multiple copies
            if new_coord in MOVE_FUNCTIONS[piece](coord, occupied_squares, player):
                
                if prev_location is not None:
                    raise ValueError(f"Multiple pieces of type {piece} can move to {new_location}.")

                prev_location = coord
        
        # Ensure we have found a valid original location
        if prev_location is None:
            raise ValueError(f"No piece of type {piece} can move to {new_location}.")

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
        player_pieces[piece].remove(prev_location)
        player_pieces[piece].append(new_coord)

        if promotion is not None:
            player_pieces[PAWN].remove(new_coord)
            player_pieces[promotion].append(new_coord)


    def castle_pieces(self, player, king_side):
        if player == WHITE and king_side:
            self.white_pieces[KING] = [COORDINATES["g1"]]
            self.white_pieces[ROOK].remove(COORDINATES["h1"])
            self.white_pieces[ROOK].append(COORDINATES["f1"])
        elif player == BLACK and king_side:
            self.black_pieces[KING] = [COORDINATES["g8"]]
            self.black_pieces[ROOK].remove(COORDINATES["h8"])
            self.white_pieces[ROOK].append(COORDINATES["f8"])
        elif player == BLACK:
            self.black_pieces[KING] = [COORDINATES["c8"]]
            self.black_pieces[ROOK].remove(COORDINATES["a8"])
            self.black_pieces[ROOK].append(COORDINATES["d8"])
        else:
            self.white_pieces[KING] = [COORDINATES["c1"]]
            self.white_pieces[ROOK].remove(COORDINATES["a1"])
            self.white_pieces[ROOK].append(COORDINATES["a8"])

    def gen_state(self):
        game_state_white = self.visible_positions(WHITE)
        game_state_black = self.visible_positions(BLACK)
        board = self.write_board()

        return game_state_white, game_state_black, board


    def visible_positions(self, player):
        positions = np.zeros((8, 8), dtype=int)

        occupied = self.occupancy()
        occupied_pieces = self.pieces()

        pieces = self.white_pieces if player == WHITE else self.black_pieces

        for piece, locations in pieces.items():

            # Uncover our Current Pieces
            for coord in locations:
                positions[coord] = piece

                # Travel to all visable positions for piece, which now stop at
                # the first piece in the way rather than seeing through it
                for move in MOVE_FUNCTIONS[piece](coord, occupied, player):
                    positions[move] = occupied_pieces.get(move, EMPTY_SQUARE)

        return positions

    def __str__(self):
        return str(self.write_board())



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
        i = 0
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
            print(f"Finished Game: {i}")
            i += 1

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

        print(move)
        print(self.state)

        # Handle the Castle
        if move == "O-O":
            self.state.castle_pieces(player, True)
        elif move == "O-O-O":
            self.state.castle_pieces(player, False)
            
        # Remove Check or Checkmate
        move = move.replace("+", "").replace("#", "")
        capture = False
        column = None
        
        # Finds the capture
        if 'x' in move:
            capture = True
            move = move.replace('x', '')

        # Promotion Handling
        if "=" in move:
            moves = move.split("=")
            new_piece = PIECES[moves[1][0]]

            print(new_piece)

            if capture is True: column = moves[0][0]

            move = moves[0][-2] + moves[0][-1]

            self.state.move_piece(player, piece, column, move, capture, new_piece)


        # Only length 4 will have a column
        if len(move) == 4:
            column = move[1]
            move = move[0] + move[2:]

        # Get the piece
        if move[0].islower():
            piece = PAWN

            # A pawn capture names the file it came from, exd5 and the en
            # passant exd6 alike, which is the hint that resolves it
            if len(move) == 3:
                column = move[0]
                move = move[1:]
        else:
            piece = PIECES[move[0]]
            move = move[1:]

        assert len(move) == 2, "Move must be in the format 'e4' or 'Nf3'"
        
        # Run move function
        self.state.move_piece(player, piece, column, move, capture, None)
        return  




        

if __name__ == "__main__":
    chessgame = ChessGame(files)

    chessgame.run_data_process()
