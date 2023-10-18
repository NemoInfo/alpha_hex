#!/usr/bin/env python3
"""MCTS Node module."""
import numpy as np
import math


class Node:
    """MCTS Node class."""

    def __init__(self, game, args, state, parent=None,
                 action_taken=None, prior=0, visit_count=0):
        """Initialize a MCTS Node."""
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        """Return `true` if node has been fully expanded."""
        return len(self.children) > 0

    def select(self):
        """Return child node with best ucb score."""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        """Compute UCB score for `child`."""
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + \
            self.args['C'] * \
            (math.sqrt(self.visit_count) / (child.visit_count + 1)) * \
            child.prior

    def expand(self, policy):
        """Expand this node according to `policy`."""
        for action, p in enumerate(policy):
            if p > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_state_perspective(child_state, player=-1)
                # Action taken to get to child, from its perspective
                child_action = self.game.change_action_perspective(action, player=-1)

                child = Node(self.game, self.args, child_state,
                             self, child_action, p)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        """Backpropagate this value "up the tree"."""
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
