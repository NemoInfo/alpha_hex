"""Hex Jax module."""
import jax
import jax.numpy as jnp
from flax.struct import dataclass
import alpha_hex.games as game

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
SIZE = 11


@dataclass
class State(game.State):
    """Class for encoding a Hex game state."""

    current_player: jnp.ndarray = jnp.int8(0)
    encoded_state: jnp.ndarray = jnp.stack((jnp.zeros(SIZE ** 2), jnp.ones(SIZE ** 2), jnp.zeros(SIZE ** 2))).astype(jnp.float32)
    terminated: jnp.ndarray = FALSE
    rewards: jnp.ndarray = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    legal_action_mask: jnp.ndarray = jnp.ones(SIZE ** 2, dtype=jnp.bool_)
    _step_count: jnp.ndarray = jnp.int32(0)
    _size: jnp.ndarray = jnp.int8(SIZE)
    # 0(black), 1(white)
    _turn: jnp.ndarray = jnp.int8(1)
    _board: jnp.ndarray = jnp.zeros(SIZE ** 2, dtype=jnp.int32)

    @property
    def game_id(self) -> game.GameId:
        """Return GameId."""
        return "hex"


class Hex(game.Game):
    """Hex game class."""

    def __init__(self):
        """Init a Hex object."""
        self.size = SIZE

    def _initial_state(self, key: jax.random.KeyArray) -> State:
        """Return an intital state from a key."""
        _, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        return State(_size=self.size, current_player=current_player)

    # TODO: The first action can also be a swap!
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
        won = self._game_ended(board, state._turn)

        return state.replace(
            current_player=1 - state.current_player,
            terminated=won,
            legal_action_mask=state.legal_action_mask.at[action].set(TRUE),
            _board=board * -1,
            _turn=1 - state._turn,
        )

    def _game_ended(self, board: jnp.ndarray, turn: jnp.ndarray) -> jnp.bool_:
        """Return `True` if `state,board` is in finnished state."""
        #     A1
        #   A2  B1
        # A3  B2  C1
        #   B3  C2
        #     C3
        #
        # if player -1:
        #   start: A1 A2 A3 (0, 3, 6)
        #     end: C1 C2 C3 (2, 5, 9)
        # if player 1:
        #   start: A1 B1 C1 (0, 1, 2)
        #     end: A3 B3 C3 (6, 7, 8)
        start, end = jax.lax.cond(
            turn == 0,
            lambda: (board[::self.size], board[self.size - 1::self.size]),
            lambda: (board[:self.size], board[-self.size:]),
        )

        # if finnish ∩ start ∩ N ≠ ∅
        return jax.vmap(
            lambda id: (id > 0) & (id == start).any()
        )(end).any()

    def _neighbours(self, action: jnp.ndarray) -> jnp.ndarray:
        """Return the neighbour actions of `action`.

        Returns `-1` for out of board neighbours.
        Outline:
                 (x,y-1)   (x-1,y)
            (x+1,y-1)  (x,y)    (x-1,y+1)
                 (x+1,y)   (x,y+1)
        """
        x, y = self.action_to_pos(action)
        xs = jnp.array([x, x+1, x+1, x, x-1, x-1])
        ys = jnp.array([y-1, y-1, y, y+1, y+1, y])
        legal = (0 <= xs) & (xs <= self.size) & (0 <= ys) & (ys <= self.size)
        return jnp.where(legal, xs * self.size + ys, -1)

    def _get_encoded_state(self, state: State, player: jnp.ndarray) -> jnp.ndarray:
        """Return state encoded in 3 channles, (-1, 0 and 1)."""
        board = state._board.reshape((self.size, self.size))
        board = jax.lax.select(player == state.current_player, board, -board)
        board_player = board > 0
        board_opponent = board < 0
        board_legal = board == 0
        ones = jnp.ones_like(board)
        turn = jax.lax.select(player == state.current_player, state._turn, 1 - state._turn)
        turn *= ones

        # NOTE: If we run parallel games the axis *might* need to be 1:
        # (e.g.) (P, 11, 11) -> (4, P, 11, 11)
        encoded_state = jnp.stack(
            [board_player, board_opponent, board_legal, turn], dtype=jnp.bool_
        )

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

    def str_to_action(self, pos_str: str) -> jnp.ndarray:
        """Return action from human-readable position."""
        pos = self.str_to_pos(pos_str)
        return self.pos_to_action(pos)

    def pretty_print_state(self, state):
        """Display the current state (nicely)."""
        no_to_symbol = {
                -1: "O",
                0: ".",
                1: "#"
        }
        mid_sum = (self.size + self.size - 2) // 2
        print("            1 A")
        for sum in range(self.size + self.size - 1):
            space_no = abs(mid_sum - sum)
            print(" " * space_no, end="")
            if sum < self.size - 1:
                if len(str(sum+2)) < 2:
                    print(" ", end="")
                print(f"{sum+2} ", end="")
            else:
                print(f" {no_to_symbol[1]} ", end="")

            for i in range(sum + 1):
                if i >= self.size:
                    break
                j = sum - i
                if j >= self.size:
                    continue
                print(no_to_symbol[jnp.sign(state._board[j * self.size + i]).item()], end=" ")
            if sum < self.size - 1:
                print(f"{chr(sum + 66)}")
            else:
                print(no_to_symbol[-1])
        print("            # 0")
