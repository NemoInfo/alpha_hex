#!/usr/bin/env python3
"""Alpha Zero parallel module."""
import numpy as np
import torch
import torch.nn.functional as F
import random
from node import Node


class SPG:
    """Self Play Game Class."""

    def __init__(self, game):
        """Initialize SPG."""
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


class MCTSParallel:
    """Parallel Monte Carlo Tree Search class."""

    def __init__(self, game, args, model):
        """Initialize MCTSParallel."""
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spgs):
        """Return probability for each action by searching through the tree."""
        policy, _ = self.model(
                torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        for i, spg in enumerate(spgs):
            valid_moves = self.game.get_valid_moves(states[i])
            policy[i] *= valid_moves
            policy[i] /= np.sum(policy[i])

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(policy[i])

        for search in range(self.args['num_searches']):
            for spg in spgs:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spgs = [map_idx for map_idx in range(len(spgs)) if spgs[map_idx].node is not None]

            if len(expandable_spgs) > 0:
                states = np.stack([spgs[map_idx].node.state for map_idx in expandable_spgs])

                policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

                for i, map_idx in enumerate(expandable_spgs):
                    node = spgs[map_idx].node

                    valid_moves = self.game.get_valid_moves(node.state)
                    policy[i] *= valid_moves
                    policy[i] /= np.sum(policy[i])

                    node.expand(policy[i])
                    node.backpropagate(value[i])


class AlphaZeroParallel:
    """Alpha Zero class."""

    def __init__(self, model, optimizer, game, args):
        """Initialize Alpha Zero."""
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        """Play self and return the games."""
        return_memory = []
        player = 1
        spgs = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spgs) > 0:
            states = np.stack([spg.state for spg in spgs])
            neutral_states = self.game.change_state_perspective(states, player)

            self.mcts.search(neutral_states, spgs)

            for i in range(len(spgs))[::-1]:
                spg = spgs[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for h_neutral_state, h_action_probs, h_player in spg.memory:
                        h_outcome = value if h_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                                self.game.get_encoded_state(h_neutral_state),
                                h_action_probs,
                                h_outcome,
                        ))
                    del spgs[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        """Train Alpha Zero on previous games."""
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        """Do train -> self play loop and save the model at each iteration."""
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for selfPlay_iteration in range(self.args["num_selfPlay_iterations"] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
