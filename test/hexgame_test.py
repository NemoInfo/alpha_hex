#!/usr/bin/env python3
"""Hexgame text module."""
from alpha_hex.hexgame import HexGame
import numpy as np


def test_neighbour_matrix():
    """Test neighbour matrix."""
    hexgame = HexGame()
    for i in range(hexgame.row_count):
        for j in range(hexgame.col_count):
            assert hexgame.neighbour_matrix[i, j] == hexgame._get_neighbours((i, j))


def test_check_win():
    """Test check win function."""
    hexgame = HexGame()

    state_win_p1 = np.array(
        [[0, 0, 0, 0, 0, 1, -1, 0, -1, -1, 0],
         [0, 0, 0, 0, 0, 1, -1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, -1, 1, -1, 0],
         [-1, 0, 0, 0, -1, -1, -1, -1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0],
         [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]]
        )

    state_win_p2 = np.array(
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         ]
        )

    assert hexgame.check_win(state_win_p1, 16)
    assert not hexgame.check_win(state_win_p1, 6)
    assert hexgame.check_win(state_win_p2, 0)
    assert not hexgame.check_win(state_win_p2, 11)
