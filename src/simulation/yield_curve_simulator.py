# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:49:19 2025

@author: ugras
"""

import numpy as np
import pandas as pd

class YieldCurveSimulator:
    """
    Simple mean-reverting yield curve simulator for multiple maturities.

    Parameters
    ----------
    maturities : list[int]
        List of maturities in months (e.g. [1,3,6,12]).
    horizon : int
        Simulation horizon in months.
    num_scenarios : int
        Number of Monte Carlo scenarios.
    initial_curve : array-like or None
        Initial yields (in percent). If None, uses a flat curve of 3.0%.
    kappa : float or array-like
        Mean-reversion speed (per month).
    theta : array-like
        Long-run mean level (in percent) for each maturity.
    sigma : float or array-like
        Volatility (in percent) per month for each maturity.
    seed : int or None
        RNG seed.
    """

    def __init__(
        self,
        maturities=[1, 3, 6, 12],
        horizon=12,
        num_scenarios=10000,
        initial_curve=None,
        kappa=0.2,
        theta=None,
        sigma=0.1,
        seed=42,
    ):
        self.maturities = list(maturities)
        self.horizon = int(horizon)
        self.num_scenarios = int(num_scenarios)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        M = len(self.maturities)
        if initial_curve is None:
            initial_curve = np.full(M, 3.0)
        self.initial_curve = np.asarray(initial_curve, dtype=float)

        # allow scalar or vector params
        self.kappa = np.full(M, kappa) if np.isscalar(kappa) else np.asarray(kappa, float)
        self.theta = np.full(M, 3.0) if theta is None else np.asarray(theta, float)
        self.sigma = np.full(M, sigma) if np.isscalar(sigma) else np.asarray(sigma, float)

        assert self.initial_curve.shape[0] == M
        assert self.kappa.shape[0] == M
        assert self.theta.shape[0] == M
        assert self.sigma.shape[0] == M

    def simulate(self):
        """
        Returns
        -------
        yields : np.ndarray
            shape (num_scenarios, horizon, num_maturities)
            yields are in percent (e.g., 3.25)
        """
        M = len(self.maturities)
        y = np.zeros((self.num_scenarios, self.horizon, M), dtype=float)
        y[:, 0, :] = self.initial_curve

        for t in range(1, self.horizon):
            prev = y[:, t - 1, :]  # shape (num_scenarios, M)
            # mean reversion term
            drift = self.kappa * (self.theta - prev)
            # Gaussian shocks independent across maturities
            shock = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.num_scenarios, M))
            y[:, t, :] = prev + drift + shock

        return y

    def to_dataframe(self, yields):
        """
        Convert yields (num_scenarios, horizon, M) to tidy DataFrame.
        """
        num_scenarios, horizon, M = yields.shape
        idx = pd.MultiIndex.from_product(
            [range(num_scenarios), range(horizon)],
            names=["scenario", "month"]
        )
        col_names = [f"{m}M" for m in self.maturities]
        df = pd.DataFrame(
            yields.reshape(num_scenarios * horizon, M),
            index=idx,
            columns=col_names
        ).reset_index()
        return df
