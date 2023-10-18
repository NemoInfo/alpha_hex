#!/usr/bin/env python3
from hexgame import HexGame
from resnet import ResNet
from alpha0.parallel import AlphaZeroParallel
import torch

game = HexGame()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'name': "marius",
    'num_searches': 300,
    'num_iterations': 8,
    'num_selfPlay_processes': 8,
    'num_selfPlay_iterations': 400,
    'num_parallel_games': 50,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()
