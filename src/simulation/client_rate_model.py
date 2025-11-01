# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 16:00:50 2025

@author: ugras
"""

import numpy as np
import pandas as pd

class ClientRateModel:
    """
    Piecewise-linear client rate model based on the market 1-month yield.

    The model maps a market reference rate (by default the 1M yield) to a
    client-facing rate using a piecewise-linear function defined by breakpoints.
    Optionally adds a reaction lag (in months) and Gaussian noise.

    Parameters
    ----------
    maturities : list[int]
        List of maturities (months) for the input yield cube, e.g. [1,3,6,12].
    breakpoints : list[float]
        Increasing x coordinates (market rate in percent) where slope/intercept changes.
        Example: [1.0, 3.0] means segments (-inf,1.0], (1.0,3.0], (3.0,inf).
    slopes : list[float]
        Slopes for each segment (length = len(breakpoints) + 1).
        Interprets client_rate = slope * market_rate + intercept (intercepts computed
        to make the piecewise function continuous).
    base_intercept : float
        Base intercept for first segment; later intercepts are computed to maintain continuity.
    lag : int
        Reaction lag in months: client_rate(t) = f(market_rate(t - lag)). If lag > 0,
        the earliest months use the earliest available market rate.
    noise_std : float
        Standard deviation of Gaussian noise (in percent) added to client rates.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        maturities=[1, 3, 6, 12],
        breakpoints=[1.0, 3.0],
        slopes=[0.5, 0.8, 1.0],
        base_intercept=0.0,
        lag=0,
        noise_std=0.0,
        seed=None,
    ):
        self.maturities = list(maturities)
        self.breakpoints = list(breakpoints)
        self.slopes = list(slopes)
        assert len(self.slopes) == len(self.breakpoints) + 1, \
            "slopes length must be breakpoints+1"
        self.base_intercept = float(base_intercept)
        self.lag = int(lag)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        # find index of 1M (or nearest) maturity
        try:
            self.ref_index = self.maturities.index(1)
        except ValueError:
            # if 1 isn't present, pick the shortest maturity and warn silently
            self.ref_index = 0

        # compute intercepts so function is continuous across segments
        self.intercepts = self._compute_intercepts()

    def _compute_intercepts(self):
        # intercepts[0] = base_intercept
        intercepts = [self.base_intercept]
        # for subsequent segments, ensure continuity:
        # slope_i * x_break + intercept_i == slope_{i-1} * x_break + intercept_{i-1}
        for i, bp in enumerate(self.breakpoints):
            prev_slope = self.slopes[i]
            next_slope = self.slopes[i + 1]
            prev_intercept = intercepts[i]
            # intercept_next = prev_slope*bp + prev_intercept - next_slope*bp
            intercepts.append(prev_slope * bp + prev_intercept - next_slope * bp)
        return np.array(intercepts, dtype=float)

    def _piecewise_map(self, x):
        """
        Vectorized piecewise linear map. x is array-like of market rates (percent).
        Returns array of client rates (percent).
        """
        x = np.asarray(x, dtype=float)
        # find segment index for each x
        # segments: (-inf, bp0], (bp0, bp1], ..., (bp_{k-1}, inf)
        idx = np.searchsorted(self.breakpoints, x, side="right")
        slopes = np.array(self.slopes)[idx]
        intercepts = self.intercepts[idx]
        return slopes * x + intercepts

    def compute(self, yields):
        """
        Compute client rates from yields.

        Parameters
        ----------
        yields : np.ndarray
            shape (num_scenarios, horizon, num_maturities), yields in percent (e.g., 3.0)

        Returns
        -------
        client_rates : np.ndarray
            shape (num_scenarios, horizon), client rates in percent
        """
        yields = np.asarray(yields, dtype=float)
        if yields.ndim != 3:
            raise ValueError("yields must be shape (num_scenarios, horizon, num_maturities)")

        ns, T, M = yields.shape
        if M != len(self.maturities):
            # allow if user passed different maturities but shapes mismatch: proceed with ref_index
            pass

        # get reference market series (1M or nearest)
        ref_series = yields[:, :, self.ref_index]  # shape (ns, T)

        # apply lag: client_rate(t) = f(ref(t - lag)); for t < lag use ref(0)
        # apply lag: client_rate(t) = f(ref(t - lag)); for t < lag use ref(0)
        if self.lag > 0:
            ref_lagged = np.zeros_like(ref_series)
            for t in range(T):
                src_t = max(0, t - self.lag)
                ref_lagged[:, t] = ref_series[:, src_t]
        else:
            ref_lagged = ref_series

        # map
        client = self._piecewise_map(ref_lagged)

        # add noise if requested
        if self.noise_std > 0.0:
            noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=client.shape)
            client = client + noise

        return client

    def to_dataframe(self, client_rates):
        """
        Convert client_rates (ns, T) to tidy DataFrame with columns: scenario, month, client_rate.
        """
        client_rates = np.asarray(client_rates, dtype=float)
        ns, T = client_rates.shape
        idx = pd.MultiIndex.from_product([range(ns), range(T)], names=["scenario", "month"])
        df = pd.DataFrame({ "client_rate": client_rates.flatten() }, index=idx).reset_index()
        return df

# Quick example / smoke test
if __name__ == "__main__":
    from simulation.yield_curve_simulator import YieldCurveSimulator  # adapt import if needed
    sim = YieldCurveSimulator(maturities=[1,3,6,12], horizon=12, num_scenarios=3, seed=123)
    y = sim.simulate()  # shape (3,12,4)
    crm = ClientRateModel(
        maturities=[1,3,6,12],
        breakpoints=[1.0, 3.0],
        slopes=[0.5, 0.8, 1.0],
        base_intercept=0.0,
        lag=1,
        noise_std=0.02,
        seed=42
    )
    cr = crm.compute(y)
    print("client rates shape:", cr.shape)
    print(cr[:2, :4])  # print first two scenarios, first 4 months
