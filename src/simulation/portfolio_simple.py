# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:39:01 2025

@author: ugras
"""

import numpy as np
from typing import Optional, Callable, Any

class PortfolioManagerSimple:
    """
    Vectorized Portfolio manager (single-path + multi-path API).

    Single-path behaviour preserved via `run_path`.
    Vectorized behaviour via `run_paths` for many scenarios.

    See docstrings of run_paths/run_path for details.
    """

    def __init__(self, initial_portfolio: np.ndarray, historical_yc: np.ndarray, maturities=None):
        """
        initial_portfolio: (hist, M) notionals for months -hist..-1  OR (S, hist, M) for per-scenario
        historical_yc: (hist, M) yields corresponding to initial_portfolio purchase yields (months -hist..-1)
                       OR (S, hist, M) for per-scenario
        maturities: list like [1,3,6,12]
        """
        # accept either 2D or 3D initial_portfolio/historical_yc
        initial = np.asarray(initial_portfolio, dtype=float)
        hist_yc = np.asarray(historical_yc, dtype=float)

        if initial.ndim == 2:
            hist_rows, cols = initial.shape
            self._per_scenario_initial = False
        elif initial.ndim == 3:
            _, hist_rows, cols = initial.shape
            self._per_scenario_initial = True
        else:
            raise ValueError("initial_portfolio must be shape (hist, M) or (S, hist, M)")

        if hist_yc.ndim not in (2, 3):
            raise ValueError("historical_yc must be shape (hist, M) or (S, hist, M)")

        if hist_yc.ndim == 2 and hist_yc.shape != (hist_rows, cols):
            raise ValueError("historical_yc shape must match initial_portfolio when 2D")
        if hist_yc.ndim == 3 and (hist_yc.shape[1] != hist_rows or hist_yc.shape[2] != cols):
            raise ValueError("historical_yc shape must match initial_portfolio when 3D")

        self.hist_rows = int(hist_rows)
        self.n_tenors = int(cols)
        self.maturities = maturities or [1, 3, 6, 12]
        if len(self.maturities) != self.n_tenors:
            raise ValueError("length of maturities must match number of columns in initial_portfolio")

        # store originals (may be broadcast later)
        self.initial_portfolio_raw = initial
        self.historical_yc_raw = hist_yc

    # ---------- small helpers ----------
    @staticmethod
    def _to_decimal_arr(arr):
        a = np.asarray(arr, dtype=float)
        return np.where(np.abs(a) > 0.5, a / 100.0, a)

    @staticmethod
    def _ensure_shapes_single(yc_scenario, cr_scenario, volume_scenario, expected_T, expected_M):
        yc = np.asarray(yc_scenario, dtype=float)
        cr = np.asarray(cr_scenario, dtype=float)
        vol = np.asarray(volume_scenario, dtype=float)
        if yc.shape != (expected_T, expected_M):
            raise ValueError(f"yc_scenario must be shape ({expected_T},{expected_M})")
        if cr.shape != (expected_T,):
            raise ValueError(f"cr_scenario must be shape ({expected_T},)")
        if vol.shape != (expected_T,):
            raise ValueError(f"volume_scenario must be shape ({expected_T},)")
        return yc, cr, vol

    @staticmethod
    def _broadcast_to_scenarios(arr, S):
        """
        If arr is (T,M) return (S,T,M) with broadcasting along axis 0.
        If arr is (S,T,M) return arr.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 2:
            T, M = a.shape
            return np.broadcast_to(a[None, :, :], (S, T, M)).copy()
        if a.ndim == 3:
            return a.copy()
        raise ValueError("array must be 2D (T,M) or 3D (S,T,M)")

    @staticmethod
    def _broadcast_to_scenarias_vector(arr, S):
        """
        If arr is (T,) return (S,T). If arr is (S,T) return arr.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            T = a.shape[0]
            return np.broadcast_to(a[None, :], (S, T)).copy()
        if a.ndim == 2:
            return a.copy()
        raise ValueError("array must be 1D (T,) or 2D (S,T)")

    # ---------- public API ----------
    def run_path(self, yc_scenario, cr_scenario, volume_scenario, strategies):
        """
        Backwards-compatible single-scenario wrapper. Calls run_paths with S=1.
        """
        # expand inputs to expected shapes and call run_paths
        yc = np.asarray(yc_scenario, dtype=float)
        cr = np.asarray(cr_scenario, dtype=float)
        vol = np.asarray(volume_scenario, dtype=float)
        if yc.ndim != 2:
            raise ValueError("yc_scenario must be shape (T,M) for single path")
        if cr.ndim != 1:
            raise ValueError("cr_scenario must be shape (T,)")
        if vol.ndim != 1:
            raise ValueError("volume_scenario must be shape (T,)")

        # build inputs for run_paths
        yc_paths = yc[None, :, :]           # (1, T, M)
        cr_paths = cr[None, :]              # (1, T)
        vol_paths = vol[None, :]            # (1, T)

        # strategies: allow (T,M) or callable. For callable we pass to run_paths as-is.
        strat_input = strategies
        if not callable(strategies):
            s_arr = np.asarray(strategies, dtype=float)
            if s_arr.ndim == 2 and s_arr.shape[0] == yc.shape[0] and s_arr.shape[1] == self.n_tenors:
                strat_input = s_arr[None, :, :]  # (1,T,M)
            elif s_arr.ndim == 1 and s_arr.shape[0] == self.n_tenors:
                strat_input = s_arr[None, :, :]  # broadcast for T months
            else:
                raise ValueError("strategies must be (T,M) or (M,) for single-path, or callable")
        # call vectorized runner
        margins, portfolios, purchase_yields, logs = self.run_paths(yc_paths, cr_paths, vol_paths, strat_input, return_logs=True)
        # unpack first scenario
        return margins[0], portfolios[0], purchase_yields[0], logs[0]

    def run_paths(self, yc_paths, cr_paths, vol_paths, strategies, return_logs: bool = False):
        """
        Vectorized runner across scenarios.

        Inputs:
          yc_paths: (S,T,M) or (T,M)
          cr_paths: (S,T) or (T,)
          vol_paths: (S,T) or (T,)
          strategies:
            - (T,M) applied to all scenarios (broadcast)
            - (S,T,M) per-scenario strategies
            - callable: strategy(s, t, portfolio_snapshot_s, yc_row_s, cr_st, vol_st)
                *Note*: callable will be invoked per-scenario inside the month loop (not vectorized).
        return_logs: if True, returns per-scenario logs (list length S). Default False (saves memory).

        Returns:
          margins: (S,T)
          portfolios: (S, total_rows, M)
          purchase_yields: (S, total_rows, M)
          logs: None or list of length S (each is list of month dicts)
        """
        # normalize arrays to 3D/2D forms
        yc_paths = np.asarray(yc_paths, dtype=float)
        cr_paths = np.asarray(cr_paths, dtype=float)
        vol_paths = np.asarray(vol_paths, dtype=float)

        # detect shapes and broadcast
        # Accept yc_paths as (T,M) or (S,T,M)
        if yc_paths.ndim == 2:
            T, M = yc_paths.shape
            S = 1
            yc_paths = yc_paths[None, :, :]
        elif yc_paths.ndim == 3:
            S, T, M = yc_paths.shape
        else:
            raise ValueError("yc_paths must be shape (T,M) or (S,T,M)")

        if M != self.n_tenors:
            raise ValueError("number of maturities in yc_paths must match initial_portfolio")

        # broadcast cr_paths and vol_paths to (S,T)
        if cr_paths.ndim == 1:
            if cr_paths.shape[0] != T:
                raise ValueError("cr_paths length must equal T")
            cr_paths = np.broadcast_to(cr_paths[None, :], (S, T)).copy()
        elif cr_paths.ndim == 2:
            if cr_paths.shape != (S, T):
                # try to broadcast if first dim is 1
                if cr_paths.shape[0] == 1 and S > 1 and cr_paths.shape[1] == T:
                    cr_paths = np.broadcast_to(cr_paths, (S, T)).copy()
                else:
                    raise ValueError("cr_paths must be shape (T,) or (S,T)")
        else:
            raise ValueError("cr_paths must be 1D or 2D")

        if vol_paths.ndim == 1:
            if vol_paths.shape[0] != T:
                raise ValueError("vol_paths length must equal T")
            vol_paths = np.broadcast_to(vol_paths[None, :], (S, T)).copy()
        elif vol_paths.ndim == 2:
            if vol_paths.shape != (S, T):
                if vol_paths.shape[0] == 1 and S > 1 and vol_paths.shape[1] == T:
                    vol_paths = np.broadcast_to(vol_paths, (S, T)).copy()
                else:
                    raise ValueError("vol_paths must be shape (T,) or (S,T)")
        else:
            raise ValueError("vol_paths must be 1D or 2D")

        # Prepare strategies:
        strat_callable = callable(strategies)
        if not strat_callable:
            s_arr = np.asarray(strategies, dtype=float)
            if s_arr.ndim == 2 and s_arr.shape == (T, M):
                # broadcast to (S,T,M)
                strategies_arr = np.broadcast_to(s_arr[None, :, :], (S, T, M)).copy()
            elif s_arr.ndim == 3 and s_arr.shape == (S, T, M):
                strategies_arr = s_arr.copy()
            elif s_arr.ndim == 1 and s_arr.shape[0] == M:
                # constant strategy per scenario and per month
                strategies_arr = np.broadcast_to(s_arr[None, None, :], (S, T, M)).copy()
            else:
                raise ValueError("strategies shape must be (T,M) or (S,T,M) or (M,)")
        else:
            strategies_arr = None  # will use callable per scenario/month

        # Prepare initial portfolio and historical_yc broadcasted to (S,hist,M)
        if self.initial_portfolio_raw.ndim == 2:
            init_port = np.broadcast_to(self.initial_portfolio_raw[None, :, :], (S, self.hist_rows, M)).copy()
        else:
            if self.initial_portfolio_raw.shape[0] != S:
                raise ValueError("initial_portfolio per-scenario first dim must match number of scenarios S")
            init_port = self.initial_portfolio_raw.copy()

        if self.historical_yc_raw.ndim == 2:
            hist_yc = np.broadcast_to(self.historical_yc_raw[None, :, :], (S, self.hist_rows, M)).copy()
        else:
            if self.historical_yc_raw.shape[0] != S:
                raise ValueError("historical_yc per-scenario first dim must match number of scenarios S")
            hist_yc = self.historical_yc_raw.copy()

        total_rows = self.hist_rows + T

        # Allocate outputs
        portfolios = np.zeros((S, total_rows, M), dtype=float)
        purchase_yields = np.zeros((S, total_rows, M), dtype=float)
        margins = np.zeros((S, T), dtype=float)
        logs = [None] * S if return_logs else None

        # fill historical portion
        portfolios[:, :self.hist_rows, :] = init_port
        purchase_yields[:, :self.hist_rows, :] = self._to_decimal_arr(hist_yc)

        # Month loop (vectorized across S)
        for t in range(T):
            row_idx = self.hist_rows + t

            # locked per scenario: sum over past active rows for each maturity
            # We'll compute for each maturity col the start index and sum across rows.
            # Initialize locked array shape (S,)
            locked = np.zeros(S, dtype=float)
            for col_idx, m in enumerate(self.maturities):
                start_s = max(0, row_idx - m + 1)
                end_s = row_idx  # exclude current
                if start_s < end_s:
                    locked += portfolios[:, start_s:end_s, col_idx].sum(axis=1)

            total_volume = vol_paths[:, t]          # shape (S,)
            available = total_volume - locked       # shape (S,)

            # strategy for this month across scenarios
            if strat_callable:
                # For each scenario we must call the callable to get its strategy vector.
                # This is slower but allows complex policies. We call per-scenario.
                strat_month = np.zeros((S, M), dtype=float)
                for s in range(S):
                    # provide portfolio snapshot for scenario s as (rows, M) up to current row
                    snapshot = portfolios[s, :row_idx, :].copy()
                    # the callable's signature is user-defined; we pass (s, t, snapshot, yc_row_s, cr_st, vol_st)
                    yc_row_s = yc_paths[s, t, :]
                    cr_st = cr_paths[s, t]
                    vol_st = vol_paths[s, t]
                    out = strategies(s, t, snapshot, yc_row_s, cr_st, vol_st)
                    out = np.asarray(out, dtype=float)
                    if out.ndim != 1 or out.size != M:
                        raise ValueError("strategy callable must return length-M vector")
                    ssum = out.sum()
                    if ssum == 0:
                        v = np.zeros(M, dtype=float)
                        v[0] = 1.0
                    else:
                        v = out / ssum
                    strat_month[s, :] = v
            else:
                # strategies_arr shape (S,T,M)
                strat_month = strategies_arr[:, t, :]  # (S,M)
                # normalize in case of small numerical drift
                sums = strat_month.sum(axis=1, keepdims=True)
                zero_mask = (sums == 0).flatten()
                if zero_mask.any():
                    strat_month[zero_mask, :] = 0.0
                    strat_month[zero_mask, 0] = 1.0
                    sums = strat_month.sum(axis=1, keepdims=True)
                strat_month = strat_month / sums

            # purchases = available[:, None] * strat_month  -> (S,M)
            purchases = available[:, None] * strat_month

            # apply purchases into portfolios
            portfolios[:, row_idx, :] += purchases

            # record purchase_yields for this row using yc_paths[s,t,:] converted to decimals
            purchase_yields[:, row_idx, :] = self._to_decimal_arr(yc_paths[:, t, :])

            # compute bond_income per scenario by summing over active rows (including current)
            bond_income = np.zeros(S, dtype=float)
            active_holdings = np.zeros((S, M), dtype=float)
            for col_idx, m in enumerate(self.maturities):
                start_s = max(0, row_idx - m + 1)
                end_s = row_idx + 1  # include current
                if start_s < end_s:
                    n_rows = portfolios[:, start_s:end_s, col_idx]    # (S, nrows)
                    y_rows = purchase_yields[:, start_s:end_s, col_idx] # (S, nrows)
                    bond_income += (n_rows * y_rows).sum(axis=1)
                    active_holdings[:, col_idx] = n_rows.sum(axis=1)

            # client cost
            cr_col = cr_paths[:, t]   # (S,)
            cr_dec = np.where(np.abs(cr_col) > 0.5, cr_col / 100.0, cr_col)
            client_cost = total_volume * cr_dec

            # margin_pct per scenario
            # handle total_volume == 0
            zero_vol_mask = (total_volume == 0.0)
            margin_pct = np.where(~zero_vol_mask, (bond_income - client_cost) / total_volume, np.nan)
            margins[:, t] = margin_pct

            # optionally build per-scenario logs (small dicts per month)
            if return_logs:
                if t == 0:
                    # initialize logs if needed
                    logs = [[] for _ in range(S)]
                for s in range(S):
                    logs[s].append({
                        "month": t,
                        "row_idx": int(row_idx),
                        "locked": float(locked[s]),
                        "available": float(available[s]),
                        "strategy": strat_month[s, :].tolist(),
                        "purchases": purchases[s, :].tolist(),
                        "active_holdings": active_holdings[s, :].tolist(),
                        "bond_income": float(bond_income[s]),
                        "client_cost": float(client_cost[s]),
                        "margin_pct": float(margin_pct[s])
                    })

        # if return_logs and we built logs variable, set logs_out
        logs_out = logs if return_logs else None
        return margins, portfolios, purchase_yields, logs_out
S