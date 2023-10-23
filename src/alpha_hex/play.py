#!/usr/bin/env python3
"""Demo of AlphaHex."""
import os
import numpy as np
import torch
from alpha_hex.hexgame import HexGame
from alpha_hex.resnet import ResNet
from alpha_hex.alpha0 import MCTS


hexgame = HexGame()
player = 1

args = {
        'C': 2,
        'num_searches': 1600,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3,
}

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(hexgame, 12, 128, device)
model.load_state_dict(torch.load("../../models/model_7_warius.pt"))
model.eval()

mcts = MCTS(hexgame, args, model)
state = hexgame.get_initial_state()

state = np.array(
    [[0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0],
     [0, 0, 0, 0, 0, 1, -1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, -1, 1, -1, 0],
     [-1, 0, 0, 0, 0, -1, -1, -1, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0],
     [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]]
)

move_history = []

while True:
    os.system('clear')
    hexgame.pretty_print_state(state)
    print("History: ", end="")
    for move in move_history:
        print(f"{move}, ", end=" ")
    print()

    if player == -1:
        action_table = hexgame.get_valid_moves(state)
        valid_actions = [i for i in range(hexgame.action_size) if action_table[i] == 1]
        valid_moves_str = list(map(lambda x: hexgame.pos_to_str(hexgame.action_to_pos(x)), valid_actions))
        print("Valid moves: ", valid_moves_str)
        move_str = input(f"{player}:")
        while move_str not in valid_moves_str:
            if move_str == "q":
                break
            print(f"'{move_str}' is not a valid move")
            print("Possible moves are: ", valid_moves_str)
            move_str = input(f"{player}:")
        if move_str == "q":
            print("quitting game ...")
            break

        pos = hexgame.str_to_pos(move_str)
        action = hexgame.pos_to_action(pos)
    else:
        neutral_state = hexgame.change_state_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)

        mcts_action_probs = [(hexgame.pos_to_str(hexgame.action_to_pos(action)), prob) for action, prob in enumerate(mcts_probs)]

        neutral_action = np.argmax(mcts_probs)
        action = hexgame.change_action_perspective(neutral_action, player)
        action = neutral_action

    action_pos = hexgame.action_to_pos(action)
    action_str = hexgame.pos_to_str(action_pos)

    move_history.append(action_str)

    state = hexgame.get_next_state(state, action, player)

    value, is_terminal = hexgame.get_value_and_terminated(state, action)

    if is_terminal:
        hexgame.pretty_print_state(state)
        print(move_history)
        hexgame.pretty_print_state(state)
        if value == 1:
            print(player, "won")
        break

    player = hexgame.get_opponent(player)
