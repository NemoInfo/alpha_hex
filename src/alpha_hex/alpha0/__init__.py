#!/usr/bin/env python3
"""Alpha Zero module."""
import numpy as np
import torch
import torch.nn.functional as F
import random
from node import Node


class MCTS:
    """Monte carlo tree search for alpha zero."""

    def __init__(self, game, args, model):
        """Initialize MCTS."""
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        """Return probability for each action by searching through the tree."""
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state), device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + \
            self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = \
                self.game.get_value_and_terminated(node.state,
                                                   node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state),
                        device=self.model.device
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

            break

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class AlphaZero:
    """Alpha Zero class."""

    def __init__(self, model, optimizer, game, args):
        """Initialize Alpha Zero."""
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        """Play self and return the games."""
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_state_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state,
                                                                    action)

            if is_terminal:
                returnMemory = []
                for h_neutral_state, h_action_probs, h_player in memory:
                    h_outcome = value if h_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                            self.game.get_encoded_state(h_neutral_state),
                            h_action_probs,
                            h_outcome,
                    ))
                    return returnMemory

            player = self.game.get_opponent(player)

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
            for selfPlay_iteration in range(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
