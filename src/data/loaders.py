# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:52:45 2025

@author: ugras
"""

import numpy as np
import pandas as pd
from pathlib import Path

def save_simulation(yields, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, yields)

def load_simulation(path):
    return np.load(path)
