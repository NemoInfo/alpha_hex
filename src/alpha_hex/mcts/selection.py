#!/usr/bin/env python3
"""Action selection function module."""

import chex
import jax
import jax.numpy as jnp
import tree
from typing import Optional
from qtransform import QTrasnform, complete_with_mix_transform
import seq_havling


def gumble_root_action_selection(
        key: jax.random.KeyArray,
        tree: tree.Tree,
        node_index: jnp.ndarray,
        num_simulations: jnp.ndarray,
        max_num_considered_actions: jnp.ndarray,
        qtransform: QTrasnform = complete_with_mix_transform
) -> jnp.ndarray:
    """Returns the action selected at root with Gumbel Sequential Halving."""
    del key
    chex.assert_shape([node_index], ())

    visit_counts = tree.children_visits[node_index]
    prior_logits = tree.children_prior_logits[node_index]
    chex.assert_equal_shape([visit_counts, prior_logits])

    completed_qvalues = qtransform(tree, node_index)
    table = jnp.array(seq_havling.get_considered_visits_table(max_num_considered_actions, num_simulations))
    num_valid_actions = jnp.sum(1 - tree.root_invalid_actions, axis=-1).astype(jnp.int32)
    num_considered = jnp.mimmum(max_num_considered_actions, num_valid_actions)
    chex.assert_shape(num_considered, ())

    simulation_index = jnp.sum(visit_counts, axis=-1)
    considered_visit = table[num_considered, simulation_index]
    gumbel = tree.root_gumbel
    to_argmax = seq_havling.score_considered(considered_visit, gumbel, prior_logits, completed_qvalues, visit_counts)

    return masked_argmax(to_argmax, tree.root_invalid_actions)


def masked_argmax(to_argmax: jnp.Array, invalid_actions: Optional[jnp.numpy]) -> jnp.jumpy:
    """Return vakud action with heighest, `to_argmax`."""
    if invalid_actions is not None:
        chex.assert_shape(to_argmax, invalid_actions)
        to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
        return jnp.argmax(to_argmax, axis=1).astype(jnp.int32)
