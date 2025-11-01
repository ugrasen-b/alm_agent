# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:51:42 2025

@author: ugras
"""

import torch
import torch.nn as nn

class SimpleAgentNet(nn.Module):
    """
    Small policy network that maps a state vector to allocation logits.
    We'll use a softmax to get allocation fractions across instruments.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ALMAgent:
    """
    Wrapper around policy network. Exposes a simple `act` function.
    """

    def __init__(self, input_dim, num_actions, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleAgentNet(input_dim=input_dim, output_dim=num_actions).to(self.device)

    def act(self, state):
        """
        state: numpy array or torch tensor (batchable)
        returns: allocation fractions that sum to 1 (numpy)
        """
        import numpy as np
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()
