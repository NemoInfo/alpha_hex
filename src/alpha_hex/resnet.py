#!/usr/bin/env python3
"""ResNet module."""

import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """ResNet class."""

    def __init__(self, game, num_resBlocks, num_hidden, device):
        """Initialize a ResNet."""
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
                nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_hidden),
                nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
                [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
                nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * game.row_count * game.col_count,
                          game.action_size)
        )

        self.valueHead = nn.Sequential(
                nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3 * game.row_count * game.col_count, 1),
                nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        """Feed `x` through the ResNet."""
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)

        return policy, value


class ResBlock(nn.Module):
    """Backbone block for ResNet."""

    def __init__(self, num_hidden):
        """Initialize a backbone block."""
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        """Feed `x` through the ResBlock."""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
