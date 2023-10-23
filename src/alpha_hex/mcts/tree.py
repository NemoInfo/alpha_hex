#!/usr/bin/env python3
"""AlphaZero MCTS Tree data structure."""

import jax.numpy as jnp
import jax
from flax.struct import dataclass
from typing import ClassVar


@dataclass
class SearchSummary:
    """MCTS search stats."""

    visit_counts: jnp.ndarray
    visit_probs: jnp.ndarray
    values: jnp.ndarray
    qvalues: jnp.ndarray


@dataclass
class Tree:
    """State of the search tree.

    Stores a batch of inputs.
    ## Notation:
    - `B` is the batch size
    - `N` is the number of nodes
    - `A` is the number of actions per node (this must be the same for every node)
    ## Fields:
    - `node_visits`: `[B, N]` visit count for each node
    - `raw_values`: `[B, N]` raw value for each node
    - `cum_value`: `[B, N]` cumulative value for each node
    - `parents`: `[B, N]` node index of parent for each node
    - `action_taken`: `[B, N]` action taken from parent to reach each node
    - `children_index`: `[B, N, A]` node index for children for each action for each node
    - `children_prior_logits`: `[B, N, A]` action prior logits of children for each action for each node
    - `children_visits`: `[B,N,A]` the vist count of children for each action for each node
    - `children_values`: `[B,N,A]` the value of the child for each action for each node (i.e. output of the value network at parent's action)
    - `children_discounts`: `[B,N,A]` the discount of the child for each action for each node
    - `states`: `[B,N,...]` the state at each node
    """

    # [B, N]
    node_visits: jnp.ndarray
    raw_values: jnp.ndarray
    cum_values: jnp.ndarray
    parents: jnp.ndarray
    action_taken: jnp.ndarray
    # [B, N, A]
    children_index: jnp.ndarray
    children_prior_logits: jnp.ndarray
    children_visits: jnp.ndarray
    children_discoutns: jnp.ndarray
    # [B, N, ...]
    states: jnp.ndarray

    # Class Constants
    ROOT_INDEX = ClassVar[int] = 0

    @property
    def num_actions(self):
        """Infer total number of actions."""
        return self.children_index.shape[-1]

    @property
    def num_simulations(self):
        """Return number of simulations so far."""
        return self.node_visits.shape[-1] - 1

    def qvalues(self, indices: jnp.ndarray):
        """Compute q-values for node indices in tree."""
        return jax.vmap(self._unbatched_qvalues)(indices)

    def _unbatched_qvalues(self, index: int) -> int:
        """Return qvalues for batch element `index`.

        Note since Alpha Zero does not use intermediate rewards, intermediate qvalues will be 0.
        """
        return self.children_visits[index] * self.children_discoutns[index]

    def summary(self) -> SearchSummary:
        """Return SearchSummary for the root node."""
        values = self.node_values[:, self.ROOT_INDEX]
        root_indices = jnp.full(values.shape, self.ROOT_INDEX)
        qvalues = self.qvalues(root_indices)

        visit_counts = self.children_visits[:, self.ROOT_INDEX].astype(values.dtype)
        total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
        visit_probs = visit_counts / jnp.maximum(total_counts, 1)
        visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)

        return SearchSummary(
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            values=values,
            qvalues=qvalues,
        )
