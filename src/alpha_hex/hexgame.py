#!/usr/bin/env python3
"""Hexgame module."""
import numpy as np


class HexGame:
    """Hexgame class."""

    def __init__(self):
        """Initialize a hexgame object."""
        self.row_count = self.col_count = 11
        self.action_size = self.row_count * self.col_count

        self.start_finnish = {
                1: [
                        {(0, j) for j in range(self.col_count)},
                        {(10, j) for j in range(self.col_count)}
                    ],
                -1: [
                        {(i, 0) for i in range(self.row_count)},
                        {(i, 10) for i in range(self.row_count)}
                    ]
        }

        self.neighbour_matrix = np.array([
            [
                self._get_neighbours((i, j)) for j in range(self.col_count)
            ] for i in range(self.row_count)
        ], dtype=set)

    def get_initial_state(self):
        """Return the starting state of the game of hex.

        This is just an empty grid
        """
        return np.zeros((self.row_count, self.col_count))

    def action_to_pos(self, action):
        """Return the grid position of an action."""
        row = action // self.col_count
        col = action % self.col_count
        return (row, col)

    def pos_to_action(self, p):
        """Return the action associated with a grid postion."""
        (row, col) = p
        return self.col_count * row + col

    def _get_neighbours(self, p):
        """Return the grid positions of neighbours."""
        (row, col) = p
        row_m1 = row - 1 >= 0
        row_p1 = row + 1 < self.row_count
        col_m1 = col - 1 >= 0
        col_p1 = col + 1 < self.col_count

        neighbours = set()
        if row_m1:
            neighbours.add((row - 1, col))
            if col_p1:
                neighbours.add((row, col + 1))
                neighbours.add((row - 1, col + 1))
        elif col_p1:
            neighbours.add((row, col + 1))

        if row_p1:
            neighbours.add((row + 1, col))
            if col_m1:
                neighbours.add((row, col - 1))
                neighbours.add((row + 1, col - 1))
        elif col_m1:
            neighbours.add((row, col - 1))

        return neighbours

    def pos_to_str(self, p):
        """Return human-readable position from grid position."""
        (row, col) = p
        row = row + 1
        col = chr(65 + col)
        return col + str(row)

    def str_to_pos(self, p_str: str):
        """Return grid position from human-readable position."""
        col = ord(p_str[0]) - 65
        row = int(p_str[1:len(p_str)]) - 1
        return (row, col)

    def get_next_state(self, state: np.ndarray, action, player):
        """Return the next state given an `action` by a `player`."""
        (row, col) = self.action_to_pos(action)
        state[row, col] = player
        return state

    def get_valid_moves(self, state: np.ndarray):
        """Return valid moves in `state`."""
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action):
        """Return `True` if action resulted in player winning."""
        if action is None:
            return False

        (row, col) = self.action_to_pos(action)
        player = state[row, col]
        assert player in [-1, 1]

        def dfs(start, visited=None):
            if visited is None:
                visited = set()
            visited.add(start)

            for next in self.neighbour_matrix[start] - visited:
                if state[next] == player:
                    dfs(next, visited)
            return visited

        for start in self.start_finnish[player][0]:
            if not state[start] == player:
                continue
            visited = dfs(start)
            if visited & self.start_finnish[player][1]:
                return True

        return False

    def get_value_and_terminated(self, state: np.ndarray, action):
        """Return `(value, is_terminated)` tuple.

        The value is 1 for a win and -1 for a loss.
        """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return -1, True
        return 0, False

    def get_opponent(self, player):
        """Return the opponent of the player, -player."""
        return -player

    def get_opponent_value(self, value):
        """Return the opponents' value, -value."""
        return -value

    def change_state_perspective(self, state: np.ndarray, player):
        """Return the `state` from the perspective of the `player`."""
        if player == 1:
            return state
        # If state is actually multiple states the row, column indexes are 1, 2
        d = int(len(state.shape) == 3)

        return state.swapaxes(0 + d, 1 + d) * player

    def change_action_perspective(self, action, player):
        """Return the `action` from the perspective of the `player`."""
        if player == 1:
            return action
        (row, col) = self.action_to_pos(action)
        return self.pos_to_action((col, row))

    def get_encoded_state(self, state: np.ndarray):
        """Return state encoded in 3 channles, (-1, 0 and 1)."""
        encoded_state = np.stack(
                (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def pretty_print_state(self, state):
        """Display the current state (nicely)."""
        no_to_symbol = {
                -1: "O",
                0: ".",
                1: "X"
        }
        mid_sum = (self.col_count + self.row_count - 2) // 2
        print("            1 A")
        for sum in range(self.col_count + self.row_count - 1):
            space_no = abs(mid_sum - sum)
            print(" " * space_no, end="")
            if sum < self.col_count - 1:
                if len(str(sum+2)) < 2:
                    print(" ", end="")
                print(f"{sum+2} ", end="")
            else:
                print("   ", end="")

            for i in range(sum + 1):
                if i >= self.row_count:
                    break
                j = sum - i
                if j >= self.col_count:
                    continue
                print(no_to_symbol[state[j, i]], end=" ")
            if sum < self.col_count - 1:
                print(f"{chr(sum + 66)}", end="")
            print()
