'''``move_white_piece``, ``move_black_piece``, ``white_promotion``, ``black_promotion``.

Making a move has to leave three things consistent: the 8x8 array, the two piece
dictionaries, and the en passant rights.  A generator reads all three, so a move
that updates only some of them produces illegal moves several plies later, a
long way from the mistake.

The individual tests below pin down one obligation each; the replays at the
bottom then play out real games and check the whole state against python-chess
after every single ply.
'''

import random

import chess
import pytest

from src import game
from tests import oracle

WHITE_EN_PASSANT = '7k/8/8/4Pp2/8/8/8/K7 w - f6 0 1'
BLACK_EN_PASSANT = '7k/8/8/8/4pP2/8/8/K7 b - f3 0 1'
WHITE_PROMOTING = '7k/4P3/8/8/8/8/8/K7 w - - 0 1'
BLACK_PROMOTING = 'K7/8/8/8/8/8/4p3/7k b - - 0 1'
WHITE_CAPTURE = '7k/8/8/3p4/8/8/8/K3R3 w - - 0 1'


def empty(board, square_):
    '''Whether a named square reads as empty on the array.'''
    return oracle.is_empty(board, oracle.at(square_))


def code_at(board, square_):
    '''The signed piece code on a named square.'''
    return oracle.code_at(board, oracle.at(square_))


def test_white_move_updates_the_array(start_board):
    '''The piece leaves the old square and arrives on the new one.'''
    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e4'), False)

    assert empty(start_board, 'e2'), 'e2 should be empty after the pawn leaves'
    assert code_at(start_board, 'e4') == oracle.code(game.WHITE, game.PAWN), (
        f'e4 should hold a white pawn, holds {code_at(start_board, "e4")!r}'
    )


def test_white_move_updates_whites_dictionary(start_board):
    '''The moving side's dictionary follows the piece.'''
    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e4'), False)

    pawns = set(start_board.white_pieces[game.PAWN])

    assert oracle.at('e4') in pawns, 'e4 should be listed among white pawns'
    assert oracle.at('e2') not in pawns, 'e2 should no longer be listed'


def test_white_move_leaves_blacks_dictionary_alone(start_board):
    '''A white move does not add anything to black's pieces.

    Both dictionaries are in scope in these methods and they are one character
    apart, so a white pawn can end up recorded as a black pawn - which then
    shows up as black having a move it does not have.
    '''
    before = {piece: set(coords) for piece, coords in start_board.black_pieces.items()}

    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e4'), False)

    after = {piece: set(coords) for piece, coords in start_board.black_pieces.items()}

    assert after == before, (
        'black pieces changed during a white move: '
        + '; '.join(
            f'piece {piece}: {oracle.names(before[piece])} -> {oracle.names(after[piece])}'
            for piece in before
            if before[piece] != after[piece]
        )
    )


def test_black_move_updates_blacks_dictionary(start_board):
    '''The same obligation, from black's side.'''
    before = {piece: set(coords) for piece, coords in start_board.white_pieces.items()}

    start_board.move_black_piece(game.PAWN, oracle.at('e7'), oracle.at('e5'), False)

    assert oracle.at('e5') in set(start_board.black_pieces[game.PAWN])
    assert oracle.at('e7') not in set(start_board.black_pieces[game.PAWN])
    assert {
        piece: set(coords) for piece, coords in start_board.white_pieces.items()
    } == before, 'white pieces changed during a black move'


def test_capture_removes_the_captured_piece():
    '''A captured piece leaves the opponent's dictionary as well as the array.

    A piece left in the dictionary keeps generating moves from a square it no
    longer occupies.
    '''
    cb, board = oracle.position(WHITE_CAPTURE)

    board.move_white_piece(game.ROOK, oracle.at('e1'), oracle.at('d5'), True)

    assert code_at(board, 'd5') == oracle.code(game.WHITE, game.ROOK)
    assert oracle.at('d5') not in set(board.black_pieces[game.PAWN]), (
        'the captured black pawn is still listed on d5'
    )


def test_double_push_gives_the_opponent_an_en_passant_right(start_board):
    '''After e2-e4 it is *black* who may capture en passant, on the pawn's square.'''
    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e4'), False)

    assert start_board.en_passant_black == oracle.at('e4'), (
        f'black should be able to take the e4 pawn, got {start_board.en_passant_black!r}'
    )
    assert start_board.en_passant_white is None, (
        'white gains no en passant right from its own move'
    )


def test_black_double_push_gives_white_an_en_passant_right(start_board):
    '''After e7-e5 it is white who may capture en passant, on e5.'''
    start_board.move_black_piece(game.PAWN, oracle.at('e7'), oracle.at('e5'), False)

    assert start_board.en_passant_white == oracle.at('e5'), (
        f'white should be able to take the e5 pawn, got {start_board.en_passant_white!r}'
    )
    assert start_board.en_passant_black is None


def test_single_push_gives_no_en_passant_right(start_board):
    '''One square is not two.'''
    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e3'), False)

    assert start_board.en_passant_black is None, (
        f'e2-e3 is not a double push, got {start_board.en_passant_black!r}'
    )


def test_en_passant_right_expires_after_one_move(start_board):
    '''The right lasts exactly one ply - a later move clears it.'''
    start_board.move_white_piece(game.PAWN, oracle.at('e2'), oracle.at('e4'), False)
    start_board.move_black_piece(game.KNIGHT, oracle.at('g8'), oracle.at('f6'), False)

    assert start_board.en_passant_black is None, (
        'the right to take on e4 should have expired once black played something else'
    )


def test_white_en_passant_capture_removes_the_bypassed_pawn():
    '''Taking en passant removes a pawn from a square the capturer never lands on.

    White captures by moving e5-f6, and the pawn that disappears is on f5.
    '''
    cb, board = oracle.position(WHITE_EN_PASSANT)

    board.move_white_piece(game.PAWN, oracle.at('e5'), oracle.at('f6'), True)

    assert code_at(board, 'f6') == oracle.code(game.WHITE, game.PAWN)
    assert empty(board, 'e5'), 'the pawn left e5'
    assert empty(board, 'f5'), 'the captured black pawn should be gone from f5'
    assert oracle.at('f5') not in set(board.black_pieces[game.PAWN]), (
        'the captured black pawn is still listed on f5'
    )


def test_black_en_passant_capture_removes_the_bypassed_pawn():
    '''Black captures by moving e4-f3, and the pawn that disappears is on f4.'''
    cb, board = oracle.position(BLACK_EN_PASSANT)

    board.move_black_piece(game.PAWN, oracle.at('e4'), oracle.at('f3'), True)

    assert code_at(board, 'f3') == oracle.code(game.BLACK, game.PAWN)
    assert empty(board, 'e4')
    assert empty(board, 'f4'), 'the captured white pawn should be gone from f4'
    assert oracle.at('f4') not in set(board.white_pieces[game.PAWN]), (
        'the captured white pawn is still listed on f4'
    )


def test_white_promotion_replaces_the_pawn():
    '''After promoting, the square holds a queen and no pawn is left listed.'''
    cb, board = oracle.position(WHITE_PROMOTING)

    board.white_promotion(oracle.at('e7'), oracle.at('e8'), game.QUEEN, False)

    assert code_at(board, 'e8') == oracle.code(game.WHITE, game.QUEEN), (
        f'e8 should hold a white queen, holds {code_at(board, "e8")!r}'
    )
    assert set(board.white_pieces[game.QUEEN]) == {oracle.at('e8')}
    assert set(board.white_pieces[game.PAWN]) == set(), (
        f'no white pawns should be left, got {oracle.names(board.white_pieces[game.PAWN])}'
    )


def test_black_promotion_replaces_the_pawn():
    '''The same, for black promoting on the first rank.'''
    cb, board = oracle.position(BLACK_PROMOTING)

    board.black_promotion(oracle.at('e2'), oracle.at('e1'), game.QUEEN, False)

    assert code_at(board, 'e1') == oracle.code(game.BLACK, game.QUEEN)
    assert set(board.black_pieces[game.QUEEN]) == {oracle.at('e1')}
    assert set(board.black_pieces[game.PAWN]) == set()


def test_promotion_can_choose_a_knight():
    '''Promotion is not always to a queen.'''
    cb, board = oracle.position(WHITE_PROMOTING)

    board.white_promotion(oracle.at('e7'), oracle.at('e8'), game.KNIGHT, False)

    assert code_at(board, 'e8') == oracle.code(game.WHITE, game.KNIGHT)
    assert set(board.white_pieces[game.KNIGHT]) == {oracle.at('e8')}


# --------------------------------------------------------------------------
# Whole-game replays
# --------------------------------------------------------------------------

def replay(seed, plies):
    '''Random legal moves, applied to a ``game.Board`` and a ``chess.Board`` together.

    Yields ``(ply, san, chess.Board, game.Board)`` after each move.  Castling is
    skipped, because ``game.Board`` has no entry point for it - see
    ``test_king_moves.test_king_may_castle_both_ways`` for that gap.
    '''
    rng = random.Random(seed)
    cb = chess.Board()
    board = oracle.board_from(cb)

    for ply in range(1, plies + 1):
        moves = [move for move in cb.legal_moves if not cb.is_castling(move)]
        if not moves:
            return

        move = rng.choice(moves)
        san = cb.san(move)

        oracle.apply(board, cb, move)
        cb.push(move)

        yield ply, san, cb, board


@pytest.mark.parametrize('seed', [1, 2, 3], ids=['game1', 'game2', 'game3'])
def test_replaying_a_game_keeps_the_position_in_step(seed):
    '''After every ply of a real game, both representations match python-chess.

    This is the end-to-end statement: whatever the move was - a quiet move, a
    capture, a double push, an en passant capture, a promotion - the array and
    the two dictionaries still describe the position python-chess describes.
    '''
    for ply, san, cb, board in replay(seed, 40):
        grid = oracle.grid_mismatches(board, cb)
        dicts = oracle.dict_mismatches(board, cb)

        assert not grid and not dicts, (
            f'ply {ply} ({san}) left the board out of step.\n'
            f'array: '
            + '; '.join(f'{sq} expected {want!r} got {got!r}' for sq, want, got in grid[:6])
            + '\ndictionaries: '
            + '; '.join(
                f'{side} piece {piece}: expected {want} got {got}'
                for side, piece, want, got in dicts[:6]
            )
        )


@pytest.mark.parametrize('seed', [1, 2, 3], ids=['game1', 'game2', 'game3'])
def test_replaying_a_game_tracks_en_passant_rights(seed):
    '''After every ply, the en passant fields say what python-chess says.'''
    for ply, san, cb, board in replay(seed, 40):
        want_white, want_black = oracle.expected_ep(cb)

        assert (board.en_passant_white, board.en_passant_black) == (want_white, want_black), (
            f'ply {ply} ({san}): expected en_passant_white='
            f'{want_white and oracle.name(want_white)} '
            f'en_passant_black={want_black and oracle.name(want_black)}, got '
            f'{board.en_passant_white and oracle.name(board.en_passant_white)} / '
            f'{board.en_passant_black and oracle.name(board.en_passant_black)}'
        )
