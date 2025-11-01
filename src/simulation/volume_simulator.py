# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 16:04:43 2025

@author: ugras
"""

import numpy as np
import pandas as pd

class VolumeSimulator:
    """
    Simulates aggregated savings account volumes across Monte Carlo scenarios.

    Parameters
    ----------
    horizon : int
        Number of months to simulate.
    num_scenarios : int
        Number of Monte Carlo scenarios.
    base_volume : float
        Base (mean) volume level (same units as output).
    trend_per_month : float
        Deterministic linear trend added each month (can be negative).
    seasonality_amplitude : float
        Amplitude of monthly seasonality (peak-to-trough ~ 2*amplitude).
    ar_coef : float
        AR(1) coefficient (0..1) controlling persistence of stochastic shocks.
    noise_std : float
        Std dev of Gaussian innovation (in same units as volume).
    jump_prob : float
        Monthly probability of a jump event.
    jump_mean : float
        Mean magnitude of jump (additive). Can be positive or negative.
    jump_std : float
        Std dev of jump size.
    seed : int | None
        RNG seed for reproducibility.
    scale_by_base : bool
        If True, seasonality and noise are scaled relative to base_volume (interpreted as fraction).
    """

    def __init__(
        self,
        horizon=12,
        num_scenarios=10000,
        base_volume=1e6,
        trend_per_month=0.0,
        seasonality_amplitude=0.05,
        ar_coef=0.8,
        noise_std=0.02,
        jump_prob=0.01,
        jump_mean=0.1,
        jump_std=0.05,
        seed=None,
        scale_by_base=True,
    ):
        self.horizon = int(horizon)
        self.num_scenarios = int(num_scenarios)
        self.base_volume = float(base_volume)
        self.trend_per_month = float(trend_per_month)
        self.seasonality_amplitude = float(seasonality_amplitude)
        self.ar_coef = float(ar_coef)
        self.noise_std = float(noise_std)
        self.jump_prob = float(jump_prob)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.scale_by_base = bool(scale_by_base)

        self.rng = np.random.default_rng(seed)

        # Precompute deterministic seasonality pattern (length = horizon)
        # Using simple monthly sinusoid with 12-month period and phase shift so month 0 isn't always peak.
        months = np.arange(self.horizon)
        self._seasonality_pattern = np.sin(2 * np.pi * months / 12.0)  # range [-1,1]

    def simulate(self):
        """
        Simulate volumes.

        Returns
        -------
        volumes : np.ndarray
            shape (num_scenarios, horizon), absolute volumes (same units as base_volume)
        """
        ns = self.num_scenarios
        T = self.horizon

        # initialize output
        vols = np.zeros((ns, T), dtype=float)

        # deterministic baseline: base + linear trend + scaled seasonality
        seasonality = self._seasonality_pattern * self.seasonality_amplitude
        if self.scale_by_base:
            seasonality = seasonality * self.base_volume

        baseline = np.array([
            self.base_volume + self.trend_per_month * t + seasonality[t]
            for t in range(T)
        ], dtype=float)  # shape (T,)

        # initialize AR state per scenario; start at baseline[0]
        state = np.full(ns, baseline[0], dtype=float)

        for t in range(T):
            # AR innovation scaled relative to base (if scale_by_base) or absolute
            if self.scale_by_base:
                noise = self.rng.normal(loc=0.0, scale=self.noise_std * self.base_volume, size=ns)
            else:
                noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=ns)

            # jump events
            jumps = np.zeros(ns, dtype=float)
            if self.jump_prob > 0:
                mask = self.rng.random(ns) < self.jump_prob
                if mask.any():
                    jumps[mask] = self.rng.normal(loc=self.jump_mean, scale=self.jump_std, size=mask.sum())
                    # interpret jump as fraction of base_volume if scaling, else absolute
                    if self.scale_by_base:
                        jumps[mask] = jumps[mask] * self.base_volume

            # AR update: state = ar_coef * state + (1 - ar_coef) * baseline[t] + noise + jumps
            state = self.ar_coef * state + (1.0 - self.ar_coef) * baseline[t] + noise + jumps

            # ensure non-negative volumes
            state = np.maximum(state, 0.0)

            vols[:, t] = state

        return vols

    def to_dataframe(self, volumes):
        """
        Convert volumes (ns, T) to tidy DataFrame with columns scenario, month, volume.
        """
        volumes = np.asarray(volumes, dtype=float)
        ns, T = volumes.shape
        idx = pd.MultiIndex.from_product([range(ns), range(T)], names=["scenario", "month"])
        df = pd.DataFrame({"volume": volumes.flatten()}, index=idx).reset_index()
        return df

if __name__ == "__main__":
    # quick smoke test
    sim = VolumeSimulator(
        horizon=12,
        num_scenarios=5,
        base_volume=1_000_000,
        trend_per_month=2000.0,
        seasonality_amplitude=0.06,
        ar_coef=0.85,
        noise_std=0.015,
        jump_prob=0.05,
        jump_mean=0.05,
        jump_std=0.03,
        seed=123,
        scale_by_base=True
    )
    v = sim.simulate()
    print("volumes shape:", v.shape)
    print(v[:3, :6])  # first 3 scenarios, first 6 months
