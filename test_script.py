# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:53:49 2025

@author: ugras
"""


from src.simulation import YieldCurveSimulator
from src.agent import ALMAgent
from src.utils import set_seed

set_seed(123)
sim = YieldCurveSimulator(horizon=12, num_scenarios=10)
y = sim.simulate()
print("yields shape:", y.shape)

agent = ALMAgent(input_dim=8, num_actions=4)
print("agent act shapes:", agent.act([0.0]*8).shape)
