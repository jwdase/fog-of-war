'''Everything at once: does the whole set of generated moves look like chess?

The per-piece modules localise a failure to one generator.  This one asks the
question the generators exist to answer - *in this position, what may this side
play?* - by dispatching over the piece dictionaries the way ``check_squares``
does, and compares the result with python-chess two ways round:

* nothing bogus: every move offered is at least a pseudo-legal chess move.
* nothing missing: every legal move in the position is offered.

The two directions are separated because they fail for different reasons.  A
generator that ignores blockers produces bogus moves; a generator that never
looks at a piece type produces missing ones.

The king is the one generator that filters for check, so the union is a mix of
pseudo-legal and legal moves.  That is consistent with both comparisons:
pseudo-legal is a superset of legal, so a legal king move list has nothing bogus
in it, and the "nothing missing" direction is measured against legal moves.
Castling is left out - ``game.Board`` has no way to express it, which
``test_king_moves`` records on its own.
'''

import chess
import pytest

from src import game
from tests import oracle

#: How to ask each piece type for its moves, mirroring ``check_squares``.
GENERATORS = {
    game.PAWN: lambda board, loc, player: board.gen_pawn_moves(loc, player),
    game.BISHOP: lambda board, loc, player: board.gen_moves_bishop(loc, player),
    game.KNIGHT: lambda board, loc, player: board.gen_moves_knight(loc, player),
    game.ROOK: lambda board, loc, player: board.gen_moves_rook(loc, player),
    game.QUEEN: lambda board, loc, player: board.gen_moves_queen(loc, player),
    game.KING: lambda board, loc, player: board.gen_king_moves(loc, player),
}


def generated_moves(board, cb, player):
    '''Every ``(origin, target)`` the generators offer ``player`` in a position.

    Castling - a king move of two columns - is dropped, to match the oracle
    below.  Whether castling is generated at all is ``test_king_moves``' job.
    '''
    moves = set()

    for piece, generate in GENERATORS.items():
        for origin in oracle.squares_of(cb, player, piece):
            for target in generate(board, origin, player):
                if piece == game.KING and abs(target[1] - origin[1]) > 1:
                    continue
                moves.add((tuple(origin), tuple(target)))

    return moves


def oracle_moves(cb, *, legal):
    '''``(origin, target)`` pairs python-chess allows, castling excluded.'''
    source = cb.legal_moves if legal else cb.generate_pseudo_legal_moves()
    return {
        (oracle.coord(m.from_square), oracle.coord(m.to_square))
        for m in source
        if not cb.is_castling(m)
    }


def describe(moves):
    '''Move pairs as sorted ``e2->e4`` strings.'''
    return sorted(f'{oracle.name(frm)}->{oracle.name(to)}' for frm, to in moves)


def test_starting_position_offers_exactly_twenty_moves(start):
    '''White has twenty moves in the starting position - sixteen pawn, four knight.

    The one position in chess whose move count everybody knows, and a quick read
    on whether the generators are in the right ballpark at all.
    '''
    cb, board = start

    moves = generated_moves(board, cb, game.WHITE)

    assert len(moves) == 20, (
        f'expected 20 opening moves, got {len(moves)}: {describe(moves)}'
    )


def test_no_bogus_moves(fen):
    '''Nothing offered is a move chess would not allow at all.

    An offered move that is not even pseudo-legal is a move through a piece, onto
    one's own piece, off the edge of the board, or in a direction the piece does
    not travel.
    '''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]

    bogus = generated_moves(board, cb, player) - oracle_moves(cb, legal=False)

    assert not bogus, f'{fen}: not chess moves - {describe(bogus)}'


def test_no_legal_move_is_missing(fen):
    '''Every move that is legal in the position is offered by some generator.'''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]

    missing = oracle_moves(cb, legal=True) - generated_moves(board, cb, player)

    assert not missing, f'{fen}: legal moves not offered - {describe(missing)}'


def test_moves_stay_on_the_board(fen):
    '''No generator hands back a coordinate that is not a square.

    Worth stating separately because an out-of-range coordinate does not raise
    when it is used to index a numpy array - a row of ``-1`` quietly reads the
    far side of the board - so a bounds mistake can hide for a long time.
    '''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]

    off_board = [
        (frm, to)
        for frm, to in generated_moves(board, cb, player)
        if not (0 <= to[0] < 8 and 0 <= to[1] < 8)
    ]

    assert not off_board, f'{fen}: off-board targets - {off_board}'


def test_no_move_lands_on_a_friendly_piece(fen):
    '''No generator offers to capture its own side's pieces.'''
    cb, board = oracle.position(fen)
    player = oracle.COLOR_FROM_CHESS[cb.turn]
    own = set(oracle.squares_of(cb, player))

    friendly_fire = [
        (frm, to) for frm, to in generated_moves(board, cb, player) if to in own
    ]

    assert not friendly_fire, (
        f'{fen}: moves onto own pieces - {describe(friendly_fire)}'
    )


def test_a_side_in_check_may_only_answer_the_check():
    '''When the king is attacked, the legal answers are the ones chess allows.

    Positions where the side to move is in check are the ones where a
    pseudo-legal generator and a legal one part company most sharply, so the
    "nothing missing" direction is worth stating on a position that is actually
    in check rather than hoping the random corpus supplies one.
    '''
    # Black king on e8 checked by a white rook down an open e-file; the only
    # legal replies are to step the king off the file or interpose the rook on it,
    # so a generator has to get both a king move and a rook move right to pass.
    cb, board = oracle.position('4k3/r7/8/8/8/8/3P1P2/4RK2 b - - 0 1')
    assert cb.is_check(), 'fixture should be a position with black in check'

    missing = oracle_moves(cb, legal=True) - generated_moves(board, cb, game.BLACK)

    assert not missing, f'legal replies to the check not offered - {describe(missing)}'
