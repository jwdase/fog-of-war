'''Recorded games in, fog-of-war training data out.

A game arrives as PGN movetext - ``'1. e4 e6 2. d4 d5 ...'`` - and has to leave
as, for every ply, what each side could see and what was actually there.

``ChessState`` is what turns one into the other, and it holds the position twice
over:

``self.rules``
    a ``chess.Board``.  It decides everything: which rook ``'Rae1'`` moves,
    whether ``'exd6'`` is an en passant capture, where the rook goes when the
    king castles, whether ``'e8=Q'`` also took a piece.  Nothing in this module
    re-implements a rule of chess or re-reads a move out of its notation.

``self.board``
    a :class:`src.game.Board` - the representation the model is trained on: an
    8x8 array of signed piece codes addressed as ``(row, column)`` with row 0 ==
    rank 8 and white negative, plus a piece dictionary per side.  It is only
    ever told what ``self.rules`` has already worked out.

:meth:`ChessState.play` steps both forward together, so a mistake in
``game.Board``'s move application surfaces as the two drifting apart - which
``tests/test_chess_state.py`` checks after every ply - rather than as quietly
wrong training data.

Coordinates, piece codes and the en passant convention all differ between the
two libraries; ``game.to_coord``, ``game.PIECE_FROM_CHESS`` and ``board_from``
below are the only places that know it.
'''

import re

import chess
import numpy as np
import pandas as pd
import torch

from main import DATA_DIR
from src import game

files = sorted(DATA_DIR.glob("*.parquet"))
output_dir = DATA_DIR / "processed"


#: Movetext tokens that are not moves: "1.", "1...", and the result at the end.
NOT_A_MOVE = re.compile(r'^(?:\d+\.*|1-0|0-1|1/2-1/2|\*)$')

#: How to ask ``game.Board`` what a piece can see.
#:
#: Every piece sees where it could move to, which is why these are the move
#: generators rather than the attack maps: a pawn sees the square in front of it,
#: which it does not attack, and sees a diagonal only when there is something
#: there to take.  The king is the exception - it is asked for the squares it
#: attacks rather than the squares it may legally move to, because what a king
#: can see does not depend on whether stepping there would walk into check.
VISION = {
    game.PAWN: lambda board, loc, player: board.gen_pawn_moves(loc, player),
    game.BISHOP: lambda board, loc, player: board.gen_moves_bishop(loc, player),
    game.KNIGHT: lambda board, loc, player: board.gen_moves_knight(loc, player),
    game.ROOK: lambda board, loc, player: board.gen_moves_rook(loc, player),
    game.QUEEN: lambda board, loc, player: board.gen_moves_queen(loc, player),
    game.KING: lambda board, loc, player: board.get_checks_king(loc),
}


def san_moves(movetext):
    '''The moves in a PGN movetext, with move numbers and the result dropped.'''
    return [token for token in movetext.split() if not NOT_A_MOVE.match(token)]


def stack_padded(samples):
    '''Per-game ``(plies, 8, 8)`` arrays as one ``(N, L, 8, 8)`` array.

    ``L`` is the longest game in the batch and the shorter ones are left as they
    were allocated - empty boards - past the end of their last real ply.
    '''
    longest = max(len(sample) for sample in samples)
    stacked = np.full((len(samples), longest, 8, 8), game.EMPTY_SQUARE, dtype=np.int8)

    for i, sample in enumerate(samples):
        stacked[i, :len(sample)] = sample

    return stacked


def piece_dicts(cb):
    '''The ``({piece code: [coord, ...]}, ...)`` pair ``game.Board`` is built from.

    Every piece code is present as a key even for a side that has none of them
    left, because ``game.Board`` reaches into these lists by key to move and
    capture and would otherwise trip over a missing one.
    '''
    built = {
        game.WHITE: {piece: [] for piece in game.PIECE_FROM_CHESS.values()},
        game.BLACK: {piece: [] for piece in game.PIECE_FROM_CHESS.values()},
    }

    for square, piece in cb.piece_map().items():
        player = game.COLOR_FROM_CHESS[piece.color]
        code = game.PIECE_FROM_CHESS[piece.piece_type]
        built[player][code].append(game.to_coord(square))

    return built[game.WHITE], built[game.BLACK]


def board_from(cb):
    '''A ``game.Board`` holding the position a ``chess.Board`` holds.

    Two things here are easy to get backwards.  ``game.Board`` takes *black*
    pieces first, and it records en passant as the square the pawn that just
    double-pushed is standing on, in the field named for the side allowed to
    capture it - where python-chess records the square that pawn skipped over,
    and only for the side to move.
    '''
    white, black = piece_dicts(cb)
    board = game.Board(black, white)

    if cb.ep_square is not None:
        if cb.turn == chess.WHITE:
            board.en_passant_white = game.to_coord(cb.ep_square - 8)
        else:
            board.en_passant_black = game.to_coord(cb.ep_square + 8)

    return board


class ChessState:
    '''One game in progress, held as both a ``chess.Board`` and a ``game.Board``.

    See the module docstring for the division of labour: python-chess rules,
    ``game.Board`` records.
    '''

    def __init__(self, fen=None):
        self.rules = chess.Board() if fen is None else chess.Board(fen)
        self.board = board_from(self.rules)

    # ----------------------------------------------------------------------
    # The position
    # ----------------------------------------------------------------------

    @property
    def turn(self):
        '''Whose move it is, as a ``game.py`` colour.'''
        return game.COLOR_FROM_CHESS[self.rules.turn]

    @property
    def white_pieces(self):
        return self.board.white_pieces

    @property
    def black_pieces(self):
        return self.board.black_pieces

    def write_board(self):
        '''The true position: one signed piece code per square.

        A copy, because callers collect these into a per-ply list and the board
        underneath goes on changing.
        '''
        return self.board.board.copy()

    # ----------------------------------------------------------------------
    # Moves
    # ----------------------------------------------------------------------

    def play(self, san):
        '''Play one move in algebraic notation - ``'Nf3'``, ``'O-O'``, ``'exd6'``.

        Returns the ``chess.Move`` it resolved to.  Raises whatever
        ``parse_san`` raises if the notation is not a legal move in this
        position, which is the point: an unplayable game is worth failing on
        rather than recording a position that never happened.
        '''
        move = self.rules.parse_san(san)
        self.push(move)
        return move

    def push(self, move):
        '''Apply an already-resolved ``chess.Move`` to both representations.

        The one place the two APIs meet.  python-chess has settled what the move
        does - which piece, from where, capture or not - and this hands that to
        the matching ``game.Board`` entry point.  ``game.Board`` picks up the
        rest from the shape of the move: the rook that follows a castling king,
        the pawn taken by an en passant capture, the castling rights a move
        gives up.
        '''
        moved = self.rules.piece_at(move.from_square)
        player = game.COLOR_FROM_CHESS[moved.color]
        capture = self.rules.is_capture(move)
        frm, to = game.to_coord(move.from_square), game.to_coord(move.to_square)

        if move.promotion:
            promoted = game.PIECE_FROM_CHESS[move.promotion]

            if player == game.WHITE:
                self.board.white_promotion(frm, to, promoted, capture)
            else:
                self.board.black_promotion(frm, to, promoted, capture)

        else:
            piece = game.PIECE_FROM_CHESS[moved.piece_type]

            if player == game.WHITE:
                self.board.move_white_piece(piece, frm, to, capture)
            else:
                self.board.move_black_piece(piece, frm, to, capture)

        self.rules.push(move)

    # ----------------------------------------------------------------------
    # The fog
    # ----------------------------------------------------------------------

    def visible_squares(self, player):
        '''The squares ``player`` can see: its own pieces, and where they may go.

        A ray stops at the first piece in its way, so a side sees the piece
        blocking it but nothing behind it - which is the whole game.
        '''
        seen = set()
        pieces = self.white_pieces if player == game.WHITE else self.black_pieces

        for piece, locations in pieces.items():
            for loc in locations:
                loc = tuple(loc)
                seen.add(loc)
                seen.update(tuple(sq) for sq in VISION[piece](self.board, loc, player))

        return seen

    def visible_positions(self, player):
        '''``player``'s view: signed piece codes where it can see, ``0`` elsewhere.

        Note that a square reads ``0`` both when it is visibly empty and when it
        is hidden; :meth:`visible_mask` is what tells those two apart.
        '''
        positions = np.full((8, 8), game.EMPTY_SQUARE, dtype=np.int8)

        for loc in self.visible_squares(player):
            positions[loc] = self.board.board[loc]

        return positions

    def visible_mask(self, player):
        '''1 on the squares ``player`` can see, 0 on the squares hidden from it.'''
        mask = np.zeros((8, 8), dtype=np.int8)

        for loc in self.visible_squares(player):
            mask[loc] = 1

        return mask

    def gen_state(self):
        '''``(white's view, black's view, the true board)`` for the current ply.'''
        return (
            self.visible_positions(game.WHITE),
            self.visible_positions(game.BLACK),
            self.write_board(),
        )

    # ----------------------------------------------------------------------
    # The names src/gui.py calls
    # ----------------------------------------------------------------------

    def get_squares(self):
        return self.write_board()

    def get_visible_squares(self, player):
        return self.visible_positions(player)

    def __str__(self):
        return str(self.rules)


class ChessGame:
    def __init__(self, files):
        self.files = iter(files)

        try:
            self.current_file = next(self.files)
        except StopIteration:
            raise ValueError("No parquet files found in the data directory.")

        self.games = self.load_data()

        # One game is one sample, held as the three boards it is made of until
        # a shard's worth have piled up.
        self.white_data = []
        self.black_data = []
        self.board_data = []

        # Shards are numbered across the whole run, not per file, so a buffer
        # that spans two parquet files cannot overwrite an earlier shard.
        self.shard = 0

    def load_data(self):
        '''Yield a list of moves for each game'''
        df = pd.read_parquet(self.current_file)

        for moves in df["moves"]:
            yield san_moves(moves)

    def run_game(self, game_moves):
        '''Play one game out, recording both points of view after every ply.

        A game is one sample, not two: white's view, black's view and the true
        board they are both a view of, all the same length and lined up ply for
        ply.  Keeping the two views together is what lets a model be handed the
        pair - or either one of them - without having to re-pair rows that were
        split apart on the way to disk.
        '''
        game_data_white = []
        game_data_black = []
        game_board_layout = []

        for i, move in enumerate(game_moves):
            player = game.WHITE if i % 2 == 0 else game.BLACK
            self.make_move(move, player)

            game_state_white, game_state_black, board_state = self.state.gen_state()

            game_data_white.append(game_state_white)
            game_data_black.append(game_state_black)
            game_board_layout.append(board_state)

        self.white_data.append(np.array(game_data_white, dtype=np.int8))
        self.black_data.append(np.array(game_data_black, dtype=np.int8))
        self.board_data.append(np.array(game_board_layout, dtype=np.int8))

    def run_data_process(self):
        i = 0
        while True:
            try:
                game_moves = next(self.games)
            except StopIteration:
                try:
                    self.current_file = next(self.files)
                    self.games = self.load_data()
                    game_moves = next(self.games)
                except StopIteration:
                    break

            self.state = ChessState()
            self.run_game(game_moves)
            print(f"Finished Game: {i}")
            i += 1

            if len(self.board_data) >= game.TENSOR_SIZE:
                self.save_data()

        # Whatever is left over is a short final shard, not something to drop
        if self.board_data:
            self.save_data()

    def save_data(self):
        '''Write one shard as four aligned tensors, one row per game.

        ``white_board``, ``black_board`` and ``correct_board`` are each
        ``(N, L, 8, 8)``: ``N`` games, ``L`` plies, an 8x8 board per ply.  Row
        ``i`` of all three is the same game, so ``white_board[i]`` and
        ``black_board[i]`` are the two views of ``correct_board[i]``.

        Games are not all the same length, so the shard is as long as its
        longest game and the shorter ones are padded out with empty boards.
        ``length`` - ``(N,)`` - says where each game really ended; the padding is
        distinguishable without it, since a real ply always has pieces on it,
        but nothing downstream should have to know that.
        '''
        output_dir.mkdir(parents=True, exist_ok=True)

        white = self.white_data[0:game.TENSOR_SIZE]
        black = self.black_data[0:game.TENSOR_SIZE]
        boards = self.board_data[0:game.TENSOR_SIZE]
        length = [len(sample) for sample in boards]

        torch.save(
            {
                "white_board" : torch.from_numpy(stack_padded(white)),
                "black_board" : torch.from_numpy(stack_padded(black)),
                "correct_board" : torch.from_numpy(stack_padded(boards)),
                "length" : torch.tensor(length, dtype=torch.int16),
            }, output_dir / f"{self.current_file.stem}-{self.shard:05d}.pt")

        self.shard += 1
        self.white_data = self.white_data[game.TENSOR_SIZE:]
        self.black_data = self.black_data[game.TENSOR_SIZE:]
        self.board_data = self.board_data[game.TENSOR_SIZE:]

    def make_move(self, move, player):
        '''Play one move for the given player.

        The notation is handed to python-chess rather than picked apart here:
        it already knows that ``'Rae1'`` names the file the rook came from,
        that ``'exd6'`` may be an en passant capture, and that ``'O-O'`` moves
        two pieces.  The ``player`` argument is now only a check that the
        movetext and the board agree about whose turn it is.
        '''
        assert player == self.state.turn, (
            f"movetext says it is {'white' if player == game.WHITE else 'black'}'s "
            f"move at {move!r}, board says otherwise"
        )

        return self.state.play(move)


if __name__ == "__main__":
    chessgame = ChessGame(files)

    chessgame.run_data_process()
