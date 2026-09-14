'''``ChessState`` - does driving ``game.Board`` from notation keep it honest?

``ChessState`` holds the position twice: a ``chess.Board`` that decides what a
move means, and a ``game.Board`` that records it in the form the model is
trained on.  Everything worth checking here is about those two agreeing, because
the failure mode is silent - a ``game.Board`` that has drifted out of step still
produces an 8x8 array of plausible-looking numbers, and it goes into the dataset
as if it were the position that was actually played.

So the replays below play real games *through the notation*, the way the data
pipeline does, and compare the whole state against python-chess after every
single ply.  That covers the moves a hand-written SAN parser gets wrong -
``'Rae1'``, ``'exd6'`` when it is en passant, ``'O-O-O'``, ``'bxa8=Q'`` - without
this file having to know which ones those are.

The fog tests are separate, and are about ``visible_positions`` being a *view*:
a side sees its own pieces and where they may go, a ray stops at the first piece
in its way, and nothing behind that piece leaks through.
'''

import random

import chess
import numpy as np
import torch
import pytest

from src import data
from src import game
from tests import oracle

#: Positions whose notation is what makes them interesting.
SAN_CASES = {
    'rook disambiguation by file': ('7k/8/8/8/8/8/4K3/R6R w - - 0 1', 'Rae1', 'a1', 'e1'),
    'rook disambiguation by rank': ('R7/8/8/8/7k/8/8/R3K3 w - - 0 1', 'R1a4', 'a1', 'a4'),
    'promotion with capture': ('1r5k/P7/8/8/8/8/8/K7 w - - 0 1', 'axb8=Q', 'a7', 'b8'),
    'white en passant': ('7k/8/8/4Pp2/8/8/8/K7 w - f6 0 1', 'exf6', 'e5', 'f6'),
    'black en passant': ('7k/8/8/8/4pP2/8/8/K7 b - f3 0 1', 'exf3', 'e4', 'f3'),
    'kingside castling': ('7k/8/8/8/8/8/8/4K2R w K - 0 1', 'O-O', 'e1', 'g1'),
    'queenside castling': ('7k/8/8/8/8/8/8/R3K3 w Q - 0 1', 'O-O-O', 'e1', 'c1'),
}


def in_step(state):
    '''``(grid problems, dictionary problems)`` between the two representations.'''
    return (
        oracle.grid_mismatches(state.board, state.rules),
        oracle.dict_mismatches(state.board, state.rules),
    )


def assert_in_step(state, where):
    '''Fail with a readable diff if ``game.Board`` has drifted from the rules board.'''
    grid, dicts = in_step(state)

    assert not grid and not dicts, (
        f'{where} left the two representations out of step.\n'
        f'array: '
        + '; '.join(f'{sq} expected {want!r} got {got!r}' for sq, want, got in grid[:6])
        + '\ndictionaries: '
        + '; '.join(
            f'{side} piece {piece}: expected {want} got {got}'
            for side, piece, want, got in dicts[:6]
        )
    )


def random_game(seed, plies):
    '''A list of SAN moves from random legal play - castling and all.'''
    rng = random.Random(seed)
    cb = chess.Board()
    moves = []

    for _ in range(plies):
        legal = list(cb.legal_moves)
        if not legal or cb.is_game_over(claim_draw=False):
            break

        move = rng.choice(legal)
        moves.append(cb.san(move))
        cb.push(move)

    return moves


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_a_new_state_is_the_starting_position():
    '''``ChessState()`` builds the position everyone starts from.'''
    state = data.ChessState()

    assert state.rules.fen() == chess.STARTING_FEN
    assert_in_step(state, 'construction')


def test_a_state_can_start_from_a_fen():
    '''An arbitrary position round-trips into both representations.'''
    fen = 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'
    state = data.ChessState(fen)

    assert_in_step(state, f'construction from {fen}')


def test_piece_dictionaries_list_every_code():
    '''Both dictionaries have all six keys, even for pieces a side has none of.

    ``game.Board`` removes a captured piece by indexing its dictionary with the
    piece code, so a side that has run out of knights still needs the key.
    '''
    state = data.ChessState('7k/8/8/8/8/8/8/K7 w - - 0 1')

    for side in (state.white_pieces, state.black_pieces):
        assert set(side) == set(game.PIECE_FROM_CHESS.values())


def test_the_true_board_is_a_copy():
    '''``write_board`` hands back a snapshot, not a window onto a moving board.

    The pipeline collects one of these per ply into a list and saves the list at
    the end; if they aliased the live array, every ply in a game would end up
    holding the final position.
    '''
    state = data.ChessState()
    before = state.write_board()

    state.play('e4')

    assert np.array_equal(before, state.write_board()) is False, (
        'the snapshot changed when the board did'
    )


# --------------------------------------------------------------------------
# Notation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('case', list(SAN_CASES), ids=list(SAN_CASES))
def test_notation_resolves_to_the_move_chess_means(case):
    '''Each awkward piece of notation moves the piece python-chess says it does.'''
    fen, san, frm, to = SAN_CASES[case]
    state = data.ChessState(fen)

    move = state.play(san)

    if frm is not None:
        assert move.from_square == chess.parse_square(frm)
        assert move.to_square == chess.parse_square(to)

    assert_in_step(state, f'{san} in {fen}')


def test_castling_moves_the_rook_too():
    '''The rook follows the king, in the array and in the dictionary.'''
    state = data.ChessState('7k/8/8/8/8/8/8/4K2R w K - 0 1')

    state.play('O-O')

    assert oracle.code_at(state.board, oracle.at('f1')) == oracle.code(game.WHITE, game.ROOK)
    assert oracle.is_empty(state.board, oracle.at('h1'))
    assert set(state.white_pieces[game.ROOK]) == {oracle.at('f1')}


def test_en_passant_removes_the_pawn_beside_the_capturer():
    '''The pawn taken en passant is not on the square the capturer lands on.'''
    state = data.ChessState('7k/8/8/4Pp2/8/8/8/K7 w - f6 0 1')

    state.play('exf6')

    assert oracle.is_empty(state.board, oracle.at('f5')), 'the captured pawn is still there'
    assert set(state.black_pieces[game.PAWN]) == set()


def test_an_illegal_move_is_refused():
    '''Notation that is not a legal move raises rather than corrupting the board.'''
    state = data.ChessState()

    with pytest.raises(ValueError):
        state.play('e5')


def test_turn_alternates():
    '''``turn`` tracks the side to move, in game.py's colours.'''
    state = data.ChessState()

    assert state.turn == game.WHITE
    state.play('e4')
    assert state.turn == game.BLACK
    state.play('e5')
    assert state.turn == game.WHITE


def test_movetext_drops_numbers_and_results():
    '''``san_moves`` keeps the moves and nothing else.'''
    assert data.san_moves('1. e4 e5 2. Nf3 Nc6 1-0') == ['e4', 'e5', 'Nf3', 'Nc6']
    assert data.san_moves('1. d4 d5 1/2-1/2') == ['d4', 'd5']
    assert data.san_moves('1. e4 *') == ['e4']


# --------------------------------------------------------------------------
# Replays - the end-to-end statement
# --------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5], ids=lambda s: f'game{s}')
def test_replaying_a_game_keeps_both_representations_in_step(seed):
    '''After every ply of a whole game, ``game.Board`` holds the real position.'''
    state = data.ChessState()

    for ply, san in enumerate(random_game(seed, 60), start=1):
        state.play(san)
        assert_in_step(state, f'ply {ply} ({san})')


@pytest.mark.parametrize('seed', [1, 2, 3], ids=lambda s: f'game{s}')
def test_replaying_a_game_tracks_en_passant_rights(seed):
    '''After every ply, the en passant fields say what python-chess says.'''
    state = data.ChessState()

    for ply, san in enumerate(random_game(seed, 60), start=1):
        state.play(san)
        want_white, want_black = oracle.expected_ep(state.rules)

        assert (state.board.en_passant_white, state.board.en_passant_black) == (
            want_white,
            want_black,
        ), f'ply {ply} ({san}) recorded the wrong en passant right'


def test_a_game_with_castling_in_it_replays():
    '''A real game, played through the notation, arrives at the real position.

    Both sides castle, a pawn is taken en route, and the whole thing goes in as
    movetext - which is the path the dataset actually takes.
    '''
    movetext = (
        '1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 Ne7 5. a3 Bxc3+ 6. bxc3 c5 7. Qg4 Qc7 '
        '8. Qxg7 Rg8 9. Qxh7 cxd4 10. Ne2 dxc3 11. f4 d4 12. h4 Nbc6 13. h5 Bd7 '
        '14. h6 O-O-O 15. Qd3 Rg6 16. h7 Rh8 17. Nxd4 Nxd4 18. Qxd4 Kb8 19. Rh3 Nf5 '
        '20. Qb4 Bc6 21. a4 Bxg2 22. Rxc3 Qb6 23. Qxb6 axb6 24. Ba3 Bxf1 25. O-O-O'
    )

    state = data.ChessState()
    reference = chess.Board()

    for san in data.san_moves(movetext):
        state.play(san)
        reference.push_san(san)
        assert_in_step(state, f'after {san}')

    assert state.rules.board_fen() == reference.board_fen()


# --------------------------------------------------------------------------
# The fog
# --------------------------------------------------------------------------

def test_a_side_always_sees_its_own_pieces(fen):
    '''Every square you have a piece on is a square you can see.'''
    state = data.ChessState(fen)

    for player in (game.WHITE, game.BLACK):
        pieces = state.white_pieces if player == game.WHITE else state.black_pieces
        own = {tuple(loc) for locations in pieces.values() for loc in locations}

        hidden = own - state.visible_squares(player)

        assert not hidden, f'{fen}: own pieces not visible - {oracle.names(hidden)}'


def test_a_view_agrees_with_the_board_where_it_can_see(fen):
    '''A visible square holds what is really on it; a hidden one reads empty.'''
    state = data.ChessState(fen)
    truth = state.write_board()

    for player in (game.WHITE, game.BLACK):
        view = state.visible_positions(player)
        seen = state.visible_squares(player)

        wrong = [
            oracle.name((row, col))
            for row in range(8)
            for col in range(8)
            if view[row, col] != (truth[row, col] if (row, col) in seen else game.EMPTY_SQUARE)
        ]

        assert not wrong, f'{fen}: view disagrees with the board at {wrong[:6]}'


def test_the_mask_marks_exactly_the_visible_squares(fen):
    '''``visible_mask`` is the fog itself, and it matches ``visible_squares``.'''
    state = data.ChessState(fen)

    for player in (game.WHITE, game.BLACK):
        mask = state.visible_mask(player)
        marked = {(row, col) for row in range(8) for col in range(8) if mask[row, col]}

        assert marked == state.visible_squares(player), f'{fen}: mask and squares differ'


def test_nothing_behind_a_blocker_is_visible():
    '''A rook sees the piece in its way and nothing past it.

    The whole point of the variant.  White's rook on a1 looks up the a-file at a
    black pawn on a5; the black rook on a8 behind it must stay hidden.
    '''
    state = data.ChessState('r6k/8/8/p7/8/8/8/R3K3 w - - 0 1')

    seen = state.visible_squares(game.WHITE)

    assert oracle.at('a5') in seen, 'the blocking pawn should be visible'
    assert oracle.at('a8') not in seen, 'the rook behind the pawn should be hidden'
    assert state.visible_positions(game.WHITE)[oracle.at('a8')] == game.EMPTY_SQUARE


def test_a_pawn_sees_forward_and_takes_sideways():
    '''Pawn vision follows pawn moves: the push, and a diagonal worth taking.

    A pawn does not attack the square in front of it but does see it, and it
    attacks both diagonals but only sees the one with something on it - so
    neither the move list nor the attack map would do on its own.
    '''
    state = data.ChessState('7k/8/8/8/8/3p4/4P3/K7 w - - 0 1')

    seen = state.visible_squares(game.WHITE)

    assert oracle.at('e3') in seen, 'the pawn should see the square it pushes to'
    assert oracle.at('e4') in seen, 'the pawn should see its double push'
    assert oracle.at('d3') in seen, 'the pawn should see the piece it can take'
    assert oracle.at('f3') not in seen, 'an empty diagonal is not visible'


# --------------------------------------------------------------------------
# What gets saved
# --------------------------------------------------------------------------

def test_stacking_pads_short_games_with_empty_boards():
    '''Games of different lengths stack into one rectangular ``(N, L, 8, 8)``.'''
    short = np.ones((3, 8, 8), dtype=np.int8)
    long_ = np.ones((5, 8, 8), dtype=np.int8)

    stacked = data.stack_padded([short, long_])

    assert stacked.shape == (2, 5, 8, 8)
    assert (stacked[0, :3] == 1).all(), 'the real plies should survive'
    assert (stacked[0, 3:] == game.EMPTY_SQUARE).all(), 'the tail should be empty boards'
    assert (stacked[1] == 1).all(), 'the longest game should not be padded at all'


def test_a_game_becomes_two_samples_against_the_same_board():
    '''One game is two training examples: white's view and black's.

    Both are paired with the same true board, and the pairing is by index - so
    a sample and its target have to line up, and both have to be as long as the
    game was.
    '''
    pipeline = data.ChessGame.__new__(data.ChessGame)
    pipeline.input_data, pipeline.output_data, pipeline.shard = [], [], 0
    pipeline.state = data.ChessState()

    moves = random_game(1, 20)
    pipeline.run_game(moves)

    assert len(pipeline.input_data) == 2, 'a game should add a sample per side'
    assert len(pipeline.output_data) == 2

    for sample, target in zip(pipeline.input_data, pipeline.output_data):
        assert len(sample) == len(moves), 'one board state per ply'
        assert len(target) == len(moves)

    assert np.array_equal(pipeline.output_data[0], pipeline.output_data[1]), (
        'both points of view answer to the same true board'
    )
    assert not np.array_equal(pipeline.input_data[0], pipeline.input_data[1]), (
        'the two points of view should not be the same picture'
    )


def test_a_saved_shard_is_n_by_l_by_8_by_8(tmp_path, monkeypatch):
    '''What lands on disk has the shape the training code expects.'''
    monkeypatch.setattr(data, 'output_dir', tmp_path)

    pipeline = data.ChessGame.__new__(data.ChessGame)
    pipeline.input_data, pipeline.output_data, pipeline.shard = [], [], 0
    pipeline.current_file = tmp_path / 'train-00000.parquet'

    lengths = [30, 44, 12]
    for seed, plies in enumerate(lengths, start=1):
        pipeline.state = data.ChessState()
        pipeline.run_game(random_game(seed, plies))

    pipeline.save_data()

    blob = torch.load(tmp_path / 'train-00000-00000.pt')
    longest = max(lengths)

    assert blob['input_data'].shape == (2 * len(lengths), longest, 8, 8)
    assert blob['output_data'].shape == (2 * len(lengths), longest, 8, 8)
    assert blob['input_data'].dtype == torch.int8
    assert blob['lengths'].tolist() == [n for n in lengths for _ in range(2)]

    # Past its own length a sample is padding, and padding is empty.
    for i, length in enumerate(blob['lengths'].tolist()):
        assert (blob['input_data'][i, length:] == game.EMPTY_SQUARE).all()
        assert (blob['output_data'][i, length:] == game.EMPTY_SQUARE).all()


def test_shards_do_not_overwrite_each_other(tmp_path, monkeypatch):
    '''Successive saves from one parquet file land in different files.

    The shard counter runs across the whole job rather than per file, because a
    buffer flushed after the input file rolled over would otherwise be named
    after the new file and collide with its first shard.
    '''
    monkeypatch.setattr(data, 'output_dir', tmp_path)

    pipeline = data.ChessGame.__new__(data.ChessGame)
    pipeline.input_data, pipeline.output_data, pipeline.shard = [], [], 0
    pipeline.current_file = tmp_path / 'train-00000.parquet'

    for seed in (1, 2):
        pipeline.state = data.ChessState()
        pipeline.run_game(random_game(seed, 10))
        pipeline.save_data()

    assert sorted(p.name for p in tmp_path.glob('*.pt')) == [
        'train-00000-00000.pt',
        'train-00000-00001.pt',
    ]


def test_a_king_sees_its_neighbours_even_in_check():
    '''Vision does not depend on legality: a king sees squares it may not enter.

    ``gen_king_moves`` filters out squares the opponent attacks, which is right
    for moves and wrong for sight - a king can see the square a rook is covering.
    '''
    # Black rook on d8 covers the whole d-file; white's king on e1 may not step
    # onto d1 or d2, but it can certainly see them.
    state = data.ChessState('3r3k/8/8/8/8/8/8/4K3 w - - 0 1')

    seen = state.visible_squares(game.WHITE)

    assert oracle.at('d1') in seen and oracle.at('d2') in seen, (
        'the king should see the squares the rook covers'
    )
    assert oracle.at('d1') not in set(state.board.gen_king_moves(oracle.at('e1'), game.WHITE)), (
        'fixture should be one where the king may not actually move there'
    )
