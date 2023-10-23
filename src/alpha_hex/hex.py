"""Hex Jax module."""
import jax
import jax.numpy as jnp
from flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
SIZE = 11


@dataclass
class State():
    """Class for encoding a Hex game state."""

    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(SIZE ** 2, dtype=jnp.bool_)
    _step_count: jnp.ndarray = jnp.int32(0)
    _size: jnp.ndarray = jnp.int8(SIZE)
    # -1(black), 1(white)
    _turn: jnp.ndarray = jnp.int8(1)
    _board: jnp.ndarray = jnp.zeros(SIZE ** 2, dtype=jnp.int32)


class Hex():
    """Hex game class."""

    def __init__(self):
        """Init a Hex object."""
        self.size = SIZE

    def _initial_state(self) -> State:
        """Return an intital state win."""
        return State(_size=self.size)

    def _next_state(self, state: State, action: jnp.ndarray) -> State:
        """Return the next state given an `action`."""
        chain_id = action + 1
        board = state._board.at[action].set(chain_id)
        neighborus = self._neighbours(action)

        def merge_chains(neighbour_idx, b):
            """Check if the neighbour at `neighbour_idx` is played by the same player and replace values in it's chain with the current `chain_id`."""
            neighbour_action = neighborus[neighbour_idx]
            return jax.lax.cond(
                (neighbour_action >= 0) & (b[neighbour_action] > 0),
                lambda: jnp.where(b == b[neighbour_action], chain_id, b),
                lambda: b,
            )

        board = jax.lax.fori_loop(0, 6, merge_chains, board)
        won = self._game_ended(state)

        return state.replace(
                       terminated=won,
                       legal_action_mask=state.legal_action_mask.at[action].set(TRUE),
                       _step_count=state._step_count + 1,
                       _board=board * -1,
                       _turn=state._turn * -1,
                       )

    def _game_ended(self, state: State) -> jnp.bool_:
        """Return `True` if `state,board` is in finnished state."""
        #     A1
        #   A2  B1
        # A3  B2  C1
        #   B3  C2
        #     C3
        #
        # if player -1:
        #   start: A1 A2 A3 (0, 3, 6)
        # finnish: C1 C2 C3 (2, 5, 9)
        # if player 1:
        #   start: A1 B1 C1 (0, 1, 2)
        #  finish: A3 B3 C3 (6, 7, 8)
        start, finnish = jax.lax.cond(
            state._turn == -1,
            lambda: (state._board[::self.size], state._board[self.size - 1::self.size]),
            lambda: (state._board[:self.size], state._board[-self.size:]),
        )

        # if finnish ∩ start ∩ N ≠ ∅
        return jax.vmap(
            lambda id: (id > 0) & (id == start).any()
        )(finnish).any()

    def _neighbours(self, action: jnp.ndarray) -> jnp.ndarray:
        """Return the neighbour actions of `action`.

        Returns `-1` for out of board neighbours.
        Outline:
                  (x,y-1)   (x+1,y-1)
            (x-1,y)    (x,y)    (x+1,y)
                (x-1,y+1)   (x,y+1)
        """
        x, y = self.action_to_pos(action)
        xs = jnp.array([x, x+1, x-1, x+1, x-1, x])
        ys = jnp.array([y-1, y-1, y, y, y+1, y+1])
        legal = (0 <= xs) & (xs <= self.size) & (0 <= ys) & (ys <= self.size)
        return jnp.where(legal, xs * self.size + ys, -1)

    def get_encoded_state(self, state: State) -> jnp.ndarray:
        """Return state encoded in 3 channles, (-1, 0 and 1)."""
        encoded_state = jnp.stack(
                (state._board < 0, state._board == 0, state._board > 0)
        ).astype(jnp.float32)

        return encoded_state

    def action_to_pos(self, action: jnp.ndarray) -> jnp.ndarray:
        """Return the grid position of an action."""
        x = action // self.size
        y = action % self.size
        return jnp.array([x, y])

    def pos_to_action(self, pos: jnp.ndarray) -> jnp.ndarray:
        """Return the action associated with a grid postion."""
        x, y = pos
        return self.size * x + y

    def pos_to_str(self, pos: jnp.ndarray) -> jnp.ndarray:
        """Return human-readable position from grid position."""
        row, col = pos
        row = row + 1
        col = chr(65 + col)
        return col + str(row)

    def str_to_pos(self, pos_str: str) -> jnp.ndarray:
        """Return grid position from human-readable position."""
        col = ord(pos_str[0]) - 65
        row = int(pos_str[1:len(pos_str)]) - 1
        return jnp.array([row, col])
