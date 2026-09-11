'''Fixtures for the board-state test suite.'''

import pytest

from src.data.data import ChessState


@pytest.fixture
def state():
    '''A freshly generated starting position.'''
    return ChessState()


@pytest.fixture
def board(state):
    '''The 8x8 piece-code array of a freshly generated starting position.'''
    return state.data
