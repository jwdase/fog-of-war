'''The coordinate system every board state is addressed through.

``data.py`` documents the intended orientation as::

    A8
            H1

so a8 is the top-left array cell and h1 the bottom-right one.
'''

import pytest

from src.data.data import COORDINATES

from .helpers import ALL_SQUARES, FILES, RANKS, square


def test_every_square_is_mapped():
    assert set(COORDINATES) == set(ALL_SQUARES)
    assert len(COORDINATES) == 64


def test_mapping_is_a_bijection_onto_the_grid():
    indices = list(COORDINATES.values())
    assert len(set(indices)) == 64
    assert set(indices) == {(row, col) for row in range(8) for col in range(8)}


@pytest.mark.parametrize(
    ("coord", "index"),
    [
        ("a8", (0, 0)),  # top-left, per the module docstring
        ("h1", (7, 7)),  # bottom-right, per the module docstring
        ("a1", (7, 0)),
        ("h8", (0, 7)),
        ("e1", (7, 4)),  # white king
        ("d8", (0, 3)),  # black queen
        ("e4", (4, 4)),
    ],
)
def test_known_squares(coord, index):
    assert COORDINATES[coord] == index


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_row_is_the_rank_counted_down_from_eight(coord):
    file_, rank = coord[0], coord[1]
    row, col = COORDINATES[coord]
    assert row == 8 - int(rank)
    assert col == FILES.index(file_)


@pytest.mark.parametrize("coord", ALL_SQUARES)
def test_decoding_an_index_round_trips(coord):
    '''The move generators decode indices with ``chr(y + ord('a')) + str(8 - x)``.

    That decode has to be the exact inverse of ``COORDINATES``, or generated
    moves will not name the squares they were computed from.
    '''
    row, col = COORDINATES[coord]
    assert square(row, col) == coord


@pytest.mark.parametrize("rank", RANKS)
def test_a_rank_shares_one_row(rank):
    rows = {COORDINATES[f + rank][0] for f in FILES}
    assert len(rows) == 1


@pytest.mark.parametrize("file_", FILES)
def test_a_file_shares_one_column(file_):
    cols = {COORDINATES[file_ + r][1] for r in RANKS}
    assert len(cols) == 1
