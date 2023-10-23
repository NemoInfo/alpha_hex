#!/usr/bin/env python3
"""Jax Hex Tests."""
import jax
import jax.numpy as jnp
import alpha_hex.games.hex as hx


def hex_neighbour_test():
    """Test neighbour function."""
    #     A1
    #   A2  B1
    # A3  B2  C1
    #   B3  C2
    #     C3

    # B2 neighbours
    hex = hx.Hex()
    B2 = hex.pos_to_action(hex.str_to_pos("B2"))
    expected_neighbours = jnp.array(list(map(
        hex.str_to_action,
        ["A2", "A3", "B3", "C2", "C1", "B1"])))
    neighbours = hex._neighbours(B2)
    assert (expected_neighbours == neighbours).all()

    # B1 neighbours
    B1 = hex.pos_to_action(hex.str_to_pos("B1"))
    expected_neighbours = jnp.array(list(map(
        hex.str_to_action,
        ["A1", "A2", "B2", "C1", "C1", "C1"]
    ))).at[-2:].set(-1)
    neighbours = hex._neighbours(B1)
    assert (expected_neighbours == neighbours).all()


def hex_win_test():
    """Test win function."""
    hex = hx.Hex()

    # Test p1 win
    state = hex.initial_state(jax.random.PRNGKey(0))
    actions_p1 = range(0, hex.size ** 2, hex.size)
    actions_p2 = range(1, hex.size ** 2, hex.size)
    for i in range(hex.size - 1):
        state = hex.next_state(state, actions_p1[i])
        state = hex.next_state(state, actions_p2[i])

    state = hex.next_state(state, actions_p1[-1])
    assert state.terminated
    state_after_terminated = hex.next_state(state, actions_p2[-1])
    assert (state_after_terminated._board == state._board).all()

    # Test p2 win
    state = hex.initial_state(jax.random.PRNGKey(0))
    actions_p1 = range(0, hex.size)
    actions_p2 = range(11, 11 + hex.size)
    for i in range(hex.size - 1):
        state = hex.next_state(state, actions_p1[i])
        state = hex.next_state(state, actions_p2[i])

    state = hex.next_state(state, actions_p1[-1])
    assert not state.terminated
    state = hex.next_state(state, actions_p2[-1])
    assert state.terminated


def hex_get_encoded_state_test():
    """Test get encoded state function."""
    hex = hx.Hex()
    state = hex.initial_state(jax.random.PRNGKey(0))
    actions_p1 = range(0, hex.size ** 2, hex.size)
    actions_p2 = range(1, hex.size ** 2, hex.size)
    for i in range(hex.size - 1):
        state = hex.next_state(state, actions_p1[i])
        state = hex.next_state(state, actions_p2[i])

    print(hex.get_encoded_state(state, 1).shape)


if __name__ == "__main__":
    hex_neighbour_test()
    hex_win_test()
    hex_get_encoded_state_test()
