#!/usr/bin/env python3
"""Time testing of AlphaHex."""
import torch
from alpha_hex.hexgame import HexGame
from alpha_hex.resnet import ResNet
import numpy as np

game = HexGame()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(game, 12, 128, device)
model.load_state_dict(torch.load("../../models/model_7_warius.pt"))
model.eval()

state = np.array(
    [[0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0],
     [0, 0, 0, 0, 0, 1, -1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, -1, 1, -1, 0],
     [-1, 0, 0, 0, 0, -1, -1, -1, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0],
     [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]]
)

game.pretty_print_state(state)


tensor = torch.tensor(game.get_encoded_state(state), device=device).unsqueeze(0)
out_policy, out_value = model(tensor)
out_value = out_value.item()
out_policy = torch.softmax(out_policy, axis=1).squeeze(0).detach().cpu().numpy()
action = np.argmax(out_policy)
pos = game.action_to_pos(action)
print(game.pos_to_str(pos))
print(out_value)
print(out_policy)
