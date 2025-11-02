# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 00:36:02 2025

@author: ugras
"""

# src/agent/env.py
"""
ALMEnv — a lightweight Gym-like environment for your ALM simulation.

API:
    env = ALMEnv(
        initial_notional,        # (hist, M) np.ndarray
        historical_yield,        # (hist, M) np.ndarray (percent or decimal)
        yc_scenario,             # (T, M) np.ndarray (percent or decimal)
        cr_scenario,             # (T,) np.ndarray
        vol_scenario,            # (T,) np.ndarray
        maturities=None,         # list of M tenors (months), default [1,3,6,12]
        stage_months=None        # None (use tenor) or scalar or list length M
    )

    state = env.reset()
    next_state, reward, done, info = env.step(action)

State vector (returned by reset/step):
    A 1-D numpy array composed of:
      [ yc_row_dec (M), cr_dec (1), total_volume (1), active_holdings (M), locked (1), available (1), t_norm(1) ]
    - yields/client rate are in decimal (e.g., 0.03)
    - t_norm is normalized time = t / (T-1) in [0,1] (helps NN learn horizon)

Action:
    - length-M array-like of non-negative numbers (will be normalized to sum 1).
    - interpreted as allocation fractions over tenors applied to `available` to compute purchases this month.

Reward:
    - monthly margin percentage (float), equal to
      (bond_income - total_volume*client_rate_dec) / total_volume
      (np.nan if total_volume == 0 -> environment returns reward = 0.0 in that case)

Notes:
    - No cash buffer or borrow modelling here — behavior matches PortfolioManagerSimpleV3 logic:
      purchases placed in current row, purchase_yield recorded from yc_scenario[t] (converted to decimal),
      bond_income computed using each holding's purchase_yield.
    - `stage_months` controls how the desired allocation is staged: purchases = (available * action) / stage_months[i].
"""
from typing import Optional, Sequence, Union, List, Tuple, Dict, Any
import numpy as np

class ALMEnv:
    def __init__(
        self,
        initial_notional: np.ndarray,
        historical_yield: np.ndarray,
        yc_scenario: np.ndarray,
        cr_scenario: np.ndarray,
        vol_scenario: np.ndarray,
        maturities: Optional[Sequence[int]] = None,
        stage_months: Optional[Union[int, Sequence[int]]] = None,
        dtype: Any = np.float32,
    ):
        # basic shape checks
        initial_notional = np.asarray(initial_notional, dtype=float)
        historical_yield = np.asarray(historical_yield, dtype=float)
        yc_scenario = np.asarray(yc_scenario, dtype=float)
        cr_scenario = np.asarray(cr_scenario, dtype=float)
        vol_scenario = np.asarray(vol_scenario, dtype=float)

        if initial_notional.ndim != 2:
            raise ValueError("initial_notional must be 2D (hist, M)")
        if historical_yield.shape != initial_notional.shape:
            raise ValueError("historical_yield must match shape of initial_notional (hist, M)")

        if yc_scenario.ndim != 2:
            raise ValueError("yc_scenario must be 2D (T, M)")
        if cr_scenario.ndim != 1:
            raise ValueError("cr_scenario must be 1D (T,)")
        if vol_scenario.ndim != 1:
            raise ValueError("vol_scenario must be 1D (T,)")

        hist, M = initial_notional.shape
        T, M2 = yc_scenario.shape
        if M != M2:
            raise ValueError("mismatch between maturities (columns) in initial_notional and yc_scenario")
        if cr_scenario.shape[0] != T or vol_scenario.shape[0] != T:
            raise ValueError("cr_scenario and vol_scenario must have length T (same as yc_scenario rows)")

        self.hist = int(hist)
        self.M = int(M)
        self.T = int(T)

        self.initial_notional = initial_notional.astype(float).copy()
        # store purchase_yield for history as decimals
        self.historical_yield = self._to_decimal_arr(historical_yield).copy()

        # forward scenarios (per-step)
        self.yc_scenario = yc_scenario.copy()
        self.cr_scenario = cr_scenario.copy()
        self.vol_scenario = vol_scenario.copy()

        self.maturities = list(maturities) if maturities is not None else [1, 3, 6, 12]
        if len(self.maturities) != self.M:
            raise ValueError("length of maturities must match number of columns in initial_notional")

        # stage months normalization
        if stage_months is None:
            self.stage_arr = np.array(self.maturities, dtype=int)
        elif np.isscalar(stage_months):
            self.stage_arr = np.full(self.M, int(stage_months), dtype=int)
        else:
            arr = np.asarray(stage_months, dtype=int)
            if arr.size != self.M:
                raise ValueError("stage_months must be scalar or length M")
            self.stage_arr = arr

        # allocate combined arrays: rows = hist + T
        self.total_rows = self.hist + self.T
        self.dtype = dtype

        # placeholders initialized in reset()
        self.portfolio = None          # shape (total_rows, M)
        self.purchase_yield = None     # shape (total_rows, M), decimals
        self.t = None                  # current step index 0..T-1
        self.row_idx = None

    @staticmethod
    def _to_decimal_arr(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        return np.where(np.abs(a) > 0.5, a / 100.0, a)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to t=0. Returns initial state (at month t=0 BEFORE applying any action).
        """
        if seed is not None:
            np.random.seed(int(seed))

        # initialise arrays
        self.portfolio = np.zeros((self.total_rows, self.M), dtype=float)
        self.purchase_yield = np.zeros((self.total_rows, self.M), dtype=float)

        # fill historical rows 0..hist-1
        self.portfolio[: self.hist, :] = self.initial_notional.copy()
        self.purchase_yield[: self.hist, :] = self.historical_yield.copy()

        self.t = 0
        self.row_idx = self.hist  # row index in arrays corresponding to t=0

        # compute initial state (before action at t=0)
        return self._get_state()

    def _get_locked_available(self, row_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute locked and available for current row index for a single scenario (returns scalars).
        locked = sum of past purchases that are still active per maturity -> summed across maturities to scalar.
        For compatibility with previous code we follow same rule: for tenor m, include rows s in [row_idx - m + 1 .. row_idx-1]
        """
        if row_idx is None:
            row_idx = self.row_idx
        locked = 0.0
        # sum for each maturity over relevant previous rows (exclude current row)
        for col_idx, m in enumerate(self.maturities):
            start_s = max(0, row_idx - m + 1)
            end_s = row_idx  # exclude current
            if start_s < end_s:
                locked += float(self.portfolio[start_s:end_s, col_idx].sum())
        total_volume = float(self.vol_scenario[self.t])
        available = total_volume - locked
        return float(locked), float(available)

    def _get_active_holdings(self, row_idx: Optional[int] = None) -> np.ndarray:
        if row_idx is None:
            row_idx = self.row_idx
        active = np.zeros(self.M, dtype=float)
        for col_idx, m in enumerate(self.maturities):
            start_s = max(0, row_idx - m + 1)
            end_s = row_idx + 1  # include current
            if start_s < end_s:
                active[col_idx] = float(self.portfolio[start_s:end_s, col_idx].sum())
        return active

    def _get_state(self) -> np.ndarray:
        """
        Build state vector for current t BEFORE action (i.e., agent sees yc[t], cr[t], vol[t], current holdings).
        Vector layout:
           [yc_row_dec (M), cr_dec (1), total_volume (1), active_holdings (M), locked (1), available (1), t_norm (1)]
        """
        yc_row = self._to_decimal_arr(self.yc_scenario[self.t])        # (M,)
        cr_dec = float(self.cr_scenario[self.t] / 100.0) if abs(self.cr_scenario[self.t]) > 0.5 else float(self.cr_scenario[self.t])
        total_volume = float(self.vol_scenario[self.t])
        active_holdings = self._get_active_holdings(self.row_idx)      # (M,)
        locked, available = self._get_locked_available(self.row_idx)
        t_norm = float(self.t / max(1, self.T - 1))
        # concatenate into 1D vector
        vec = np.concatenate([
            yc_row.astype(float),
            np.array([cr_dec, total_volume], dtype=float),
            active_holdings.astype(float),
            np.array([locked, available, t_norm], dtype=float)
        ]).astype(self.dtype)
        return vec

    def step(self, action: Union[np.ndarray, Sequence[float]]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply action for current month t, advance environment by one month.

        action: length-M array-like (non-negative numbers). Will be normalized to sum=1.
        Returns: next_state, reward (float), done (bool), info (dict)
        """
        if self.t is None:
            raise RuntimeError("Call reset() before step()")

        # capture current state vector shape safely (call _get_state while self.t is valid)
        current_state = self._get_state()
        state_shape = current_state.shape

        a = np.asarray(action, dtype=float)
        if a.ndim != 1 or a.size != self.M:
            raise ValueError(f"action must be length-{self.M} vector")
        # non-negativity & normalization
        a = np.clip(a, 0.0, None)
        s = a.sum()
        if s == 0.0:
            # default to put all in shortest tenor (index 0)
            alloc = np.zeros_like(a); alloc[0] = 1.0
        else:
            alloc = a / s

        # compute locked/available for this month
        locked, available = self._get_locked_available(self.row_idx)

        # compute desired absolute allocation and stage per tenor
        desired = available * alloc  # absolute desired allocation
        monthly_purchase = np.zeros_like(desired)
        for i in range(self.M):
            k = max(1, int(self.stage_arr[i]))
            monthly_purchase[i] = desired[i] / k

        # place purchases into portfolio at current row
        self.portfolio[self.row_idx, :] += monthly_purchase

        # record purchase_yield for the current row using yc_scenario[t] -> decimal
        self.purchase_yield[self.row_idx, :] = self._to_decimal_arr(self.yc_scenario[self.t])

        # compute bond income for this month using purchase_yield from each row (only active rows)
        bond_income = 0.0
        active_holdings = np.zeros(self.M, dtype=float)
        for col_idx, m in enumerate(self.maturities):
            start_s = max(0, self.row_idx - m + 1)
            end_s = self.row_idx + 1  # include current
            if start_s < end_s:
                n_rows = self.portfolio[start_s:end_s, col_idx]
                y_rows = self.purchase_yield[start_s:end_s, col_idx]
                bond_income += float((n_rows * y_rows).sum())
                active_holdings[col_idx] = float(n_rows.sum())

        # client cost
        cr_val = float(self.cr_scenario[self.t])
        cr_dec = cr_val / 100.0 if abs(cr_val) > 0.5 else cr_val
        client_cost = float(self.vol_scenario[self.t]) * cr_dec

        # margin percentage reward
        total_volume = float(self.vol_scenario[self.t])
        if total_volume == 0.0:
            reward = 0.0
            margin_pct = float("nan")
        else:
            margin_pct = (bond_income - client_cost) / total_volume
            reward = float(margin_pct)

        info = {
            "t": int(self.t),
            "row_idx": int(self.row_idx),
            "locked": locked,
            "available": available,
            "monthly_purchase": monthly_purchase.tolist(),
            "active_holdings": active_holdings.tolist(),
            "bond_income": bond_income,
            "client_cost": client_cost,
            "margin_pct": margin_pct
        }

        # determine done BEFORE advancing time
        done = (self.t >= self.T - 1)

        # advance time indices only if not done
        if not done:
            self.t += 1
            self.row_idx += 1
            next_state = self._get_state()
        else:
            # terminal: return a zero-state (same shape as state) without calling _get_state()
            next_state = np.zeros(state_shape, dtype=self.dtype)

        return next_state, reward, done, info


        # advance time
        done = (self.t >= self.T - 1)
        self.t += 1
        self.row_idx += 1

        # next state: if done, return last state (or zeros) — we'll return final state's vector
        if not done:
            next_state = self._get_state()
        else:
            # At termination, produce a final state (all zeros) or the last state after step.
            # We'll return the current active snapshot (post-step) but t points beyond last month.
            # Build a terminal state of zeros with same dimension
            next_state = np.zeros_like(self._get_state())

        return next_state, reward, done, info

    def seed(self, seed: int):
        np.random.seed(int(seed))

    def render(self):
        """Minimal render: print summary of current row (useful for debugging)."""
        locked, available = self._get_locked_available(self.row_idx)
        print(f"t={self.t}, row={self.row_idx}, locked={locked:.2f}, available={available:.2f}")
        print("Current holdings (per tenor):", self._get_active_holdings(self.row_idx).tolist())
