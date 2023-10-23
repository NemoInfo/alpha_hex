#!/usr/bin/env python3
"""Abstract State and Game module."""

import abc
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Tuple, Literal, get_args

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

GameId = Literal[
    "hex",
]


def make(id: GameId):
    """Load game by id."""
    if id == "hex":
        from alpha_hex.games.hex import Hex

        return Hex()
    else:
        games = "\n".join(available_games())
        raise ValueError(
            f"Wrong Game Id, got `{id}`. Available games are:\n {games}"
        )


def available_games():
    """Return a list of available games."""
    return get_args(GameId)


@dataclass
class State(abc.ABC):
    """Abstract Game State Class."""

    current_player: jnp.ndarray
    encoded_state: jnp.ndarray
    terminated: jnp.ndarray
    rewards: jnp.ndarray
    legal_action_mask: jnp.ndarray
    _step_count: jnp.ndarray

    @property
    @abc.abstractmethod
    def game_id(self) -> GameId:
        """Game Id e.g. "hex"."""
        ...


class Game(abc.ABC):
    """Abstract Game Class."""

    @abc.abstractmethod
    def __init__(self):
        """Override."""
        ...

    @abc.abstractmethod
    def _initial_state(self, key: jax.random.KeyArray) -> State:
        """Return the initial state for this game."""
        ...

    @abc.abstractmethod
    def _get_encoded_state(self, state: State, player: jnp.ndarray) -> State:
        """Return the encoded state given `state`."""
        ...

    def encoded_state_shape(self) -> Tuple[int, ...]:
        """Return the shape of the encoded state."""
        state = self.init(jax.random.PRNGKey(0))
        encoded_state = self._get_encoded_state(state, state.current_player)
        return encoded_state.shape

    @abc.abstractmethod
    def _next_state(self, state: State, action: jnp.ndarray) -> State:
        """Return the next state for this game, given current `state` and `action`."""
        ...

    def get_encoded_state(self, state: State, player: jnp.ndarray) -> jnp.ndarray:
        """Return the encoded state given `state` and stop gradient."""
        encoded_state = self._get_encoded_state(state, player)
        return jax.lax.stop_gradient(encoded_state)

    def initial_state(self, key: jax.random.KeyArray) -> State:
        """Return the initial state for this game API."""
        state = self._initial_state(key)
        encoded_state = self.get_encoded_state(state, state.current_player)
        return state.replace(encoded_state=encoded_state)

    def next_state(self, state: State, action: jnp.ndarray) -> State:
        """Return the next step for this game API, given current `state` and `action`.

        This method handles terminal states.
        """
        assert state.legal_action_mask[action]

        state: State = jax.lax.cond(
            state.terminated,
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._next_state(state.replace(_step_count=state._step_count + 1), action),
        )

        encoded_state = self.get_encoded_state(state, state.current_player)

        return state.replace(encoded_state=encoded_state)
