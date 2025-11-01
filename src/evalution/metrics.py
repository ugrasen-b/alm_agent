# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:53:00 2025

@author: ugras
"""

import numpy as np

def compute_portfolio_return(allocations, yields):
    """
    allocations: (T, N_instruments) fractions (sum to 1 each period)
    yields: (T, N_instruments) yields in percent (e.g., 3.0)
    returns: scalar cumulative return (simple approximation)
    """
    # convert yields percent -> decimal
    r = yields / 100.0
    # period returns = dot(alloc, r)
    period_returns = (allocations * r).sum(axis=1)
    # cumulative product of (1 + r) - 1
    cumulative = np.prod(1.0 + period_returns) - 1.0
    return cumulative

def compute_sharpe(returns, risk_free=0.0):
    # returns is series of period returns
    ex = returns - risk_free
    if ex.std() == 0:
        return np.nan
    return ex.mean() / ex.std() * np.sqrt(12)  # annualized if monthly
