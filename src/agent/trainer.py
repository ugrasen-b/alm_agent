# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:52:16 2025

@author: ugras
"""

# very small trainer skeleton to show how you'd train with supervised or RL loop
import torch
import torch.optim as optim

class Trainer:
    def __init__(self, agent, lr=1e-3):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        # placeholder: iterate over batches and update model
        self.agent.model.train()
        total_loss = 0.0
        for batch in dataloader:
            # user to define: batch -> state, target/allocation or reward signals
            state, target = batch
            logits = self.agent.model(state)
            loss = ((logits - target) ** 2).mean()  # placeholder MSE
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss
