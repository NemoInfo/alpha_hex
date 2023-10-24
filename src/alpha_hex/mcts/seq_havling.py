#!/usr/bin/env python3
"""Sequetial halving functions module."""

import math
import jax.numpy as jnp
import chex


def score_considered(considered_visit, gumbel, logits, normalized_qvalues, visit_counts):
    """Return score that can be argmaxed."""
    low_logits = -1e9
    logits -= jnp.max(logits, keepdims=True, axis=-1)
    penalty = jnp.where(visit_counts == considered_visit, 0, -jnp.inf)
    chex.assert_equal_shape([gumbel, logits, normalized_qvalues, penalty])
    return jnp.maximum(low_logits, gumbel + logits + normalized_qvalues) + penalty


def get_considered_visits_table(max_num_considered_actions, num_simulations):
    """Return a table of sequences of visit counts."""
    return tuple(
        get_considered_visits_sequence(m, num_simulations)
        for m in range(max_num_considered_actions + 1)
    )


def get_considered_visits_sequence(max_num_considered_actions, num_simulations):
    """Return a Sequential Halving visit counts sequence."""
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visists = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, num_simulations // (log2max * num_considered))
        for _ in range(num_extra_visits):
            sequence += visists[:num_considered]
            for i in range(num_considered):
                visists[i] += 1

        num_considered = max(2, num_considered // 2)

    return tuple(sequence[:num_simulations])
