#!/usr/bin/env python3
"""Monotonic transform for Q-values module."""
import tree
import jax
import jax.numpy as jnp
from typing import Callable

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

QTrasnform = Callable[[tree.Tree, jnp.ndarray], jnp.ndarray]


def complete_with_mix_transform(
        tree: tree.Tree,
        node_idx: jnp.ndarray,
        value_scale: jnp.ndarray = jnp.float32(0.1),
        maxvisit_init: jnp.ndarray = jnp.float32(50.0),
        epsilon: jnp.ndarray = jnp.float32(1e-8),

) -> jnp.ndarray:
    """Return completed Q-Values.

    Missing Q-Values are replaced by the mixed value.
    [Policy improvement by planning with Gumbel, Appendix D](https://openreview.net/pdf?id=bERaNdoegnO#nameddest=Mixed%20Value%20Approximation)
    """
    qvalues = tree.qvalues(node_idx)
    visit_counts = tree.children_visits[node_idx]

    raw_value = tree.raw_values[node_idx]
    prior_probs = jax.nn.softmax(tree.children_prior_logits[node_idx])

    sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
    sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)

    weighted_q = jnp.sum(jnp.where(
        visit_counts > 0,
        prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
        0.0, axis=-1
    ))

    value = (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)

    completed_qvalues = jnp.where(visit_counts > 0, qvalues, value)

    min_value = jnp.min(qvalues, axis=-1, keepdims=True)
    max_value = jnp.max(qvalues, axis=-1, keepdims=True)

    completed_qvalues = (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)

    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit

    return visit_scale * value_scale * completed_qvalues
