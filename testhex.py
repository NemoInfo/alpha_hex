#!/usr/bin/env python3
"""Usless module."""
import numpy as np
import torch
from hexgame import HexGame
from resnet import ResNet

import matplotlib.pyplot as plt

hexgame = HexGame()
device = ("cuda" if torch.cuda.is_available() else "cpu")

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

encoded_state = hexgame.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(hexgame, 12, 128, device)
model.load_state_dict(torch.load("model_7.pt"))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

plt.bar(range(hexgame.action_size), policy)
plt.show()

action = np.argmax(policy)
p = hexgame.action_to_pos(action)
print(hexgame.pos_to_str(p))
print(value)
hexgame.pretty_print_state(state)
state = hexgame.get_next_state(state, action, 1)
hexgame.pretty_print_state(state)
