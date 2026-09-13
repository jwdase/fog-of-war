'''Shared fixtures for the ``src/game.py`` test suite.

The suite is built around one idea: ``python-chess`` decides what is legal, and
``src/game.py`` has to agree with it.  See ``tests/oracle.py`` for the bridge
between the two.
'''

import chess
import pytest

from tests import oracle

START_FEN = chess.STARTING_FEN


@pytest.fixture
def start():
    '''An ``(oracle, subject)`` pair for the starting position.'''
    return oracle.position(START_FEN)


@pytest.fixture
def start_board(start):
    '''A ``game.Board`` holding the starting position.'''
    return start[1]


def pytest_generate_tests(metafunc):
    '''Feed ``fen`` parameters from random legal play, labelled by seed and ply.

    Any test that takes a ``fen`` argument gets swept across the same corpus of
    midgame positions, so the per-piece tests and the whole-position tests are
    talking about the same chess.
    '''
    if 'fen' in metafunc.fixturenames:
        corpus = oracle.random_positions()
        metafunc.parametrize(
            'fen',
            [fen for _, fen in corpus],
            ids=[label for label, _ in corpus],
        )
