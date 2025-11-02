# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 16:28:41 2025

@author: ugras
"""

# src/simulation/strategy.py
from typing import Dict, List, Tuple
import numpy as np

# Assumes PortfolioManager and BondHolding from portfolio.py are available
# If PortfolioManager lives elsewhere, adjust the import path accordingly:
# from src.simulation.portfolio import PortfolioManager, BondHolding

def _to_decimal(x):
    """Convert percent (e.g. 3.0) to decimal (0.03).
    If input already looks like decimal (< 0.5), keep as is."""
    x = float(x)
    return x / 100.0 if abs(x) > 0.5 else x

class StandardRolloverExecutor:
    """
    Execute 'standard_rollover' strategy:
      - strategy: dict mapping maturity_months -> fraction (sums to 1)
      - available_for_invest = total_volume - locked_amount
      - desired allocation = available_for_invest * strategy_fraction
      - but actual purchase this month for maturity K is (desired_allocation / K)
        (i.e., staged equally across K months)
      - if insufficient cash to execute all monthly installments, we proportionally scale down;
      - any un-executed desired notional is reallocated to shortest tenor (1M) and executed to the extent possible.
    """

    def __init__(self, maturities: List[int] = [1, 3, 6, 12]):
        # canonical maturities order
        self.maturities = list(maturities)
        if 1 not in self.maturities:
            # ensure 1M exists as fallback
            self.maturities = [1] + self.maturities

    def execute(
        self,
        portfolio,                 # PortfolioManager instance
        strategy: Dict[int, float],# e.g. {1:0.0, 3:0.6, 6:0.0, 12:0.4}
        total_volume: float,       # total_volume at current month (absolute units)
        current_yields: Dict[int, float],  # map maturity->yield (percent or decimal)
        client_rate: float,        # client rate (percent or decimal) for the month
        allow_borrow: bool = False,
        borrow_amount: float = 0.0,
    ) -> Tuple[Dict, float]:
        """
        Execute one-month purchases according to the standard_rollover policy.

        Returns:
          executed_summary: dict with executed purchases list and final cash buffer info
          margin: float (monthly margin according to simplified formula)
        """
        # 1) compute locked and available_by_volume
        locked_amount = portfolio.compute_locked_amount()
        available_by_volume = max(0.0, total_volume - locked_amount)

        # We use available_by_volume as the investable pool (per your rule).
        investable = available_by_volume

        # If portfolio also tracks cash_buffer and you prefer using that, consider mixing:
        # investable = min(investable, portfolio.cash_buffer)  # optional

        # Optional borrowing
        if allow_borrow and borrow_amount > 0.0:
            portfolio.borrow(borrow_amount)
            investable += borrow_amount

        # 2) Normalize strategy: ensure keys match maturities and sum to 1
        # Fill missing maturities with 0
        strat = {m: float(strategy.get(m, 0.0)) for m in self.maturities}
        total_frac = sum(strat.values())
        if total_frac <= 0:
            # nothing to do
            return {"executed": [], "note": "zero strategy"}, 0.0
        # normalize fractions to sum to 1
        strat = {m: v / total_frac for m, v in strat.items()}

        # 3) Desired notional per maturity (absolute)
        desired = {m: strat[m] * investable for m in self.maturities}

        # 4) For each maturity m, the monthly installment to buy now is desired[m] / m
        monthly_targets = {m: desired[m] / max(1, m) for m in self.maturities}

        # 5) Try to execute monthly_targets using available cash = portfolio.cash_buffer + any inflows
        # We'll use portfolio.cash_buffer as the actionable cash pool.
        actionable_cash = portfolio.cash_buffer + 0.0

        executed = []
        total_demand = sum(monthly_targets.values())

        # If actionable_cash < total_demand, scale down proportionally
        # (we could also scale per-priority; user asked proportional scaling)
        if actionable_cash < total_demand and total_demand > 0:
            scale = actionable_cash / total_demand
        else:
            scale = 1.0

        # execute in maturity order (largest tenors first or any order you prefer)
        for m in sorted(self.maturities, reverse=True):
            target = monthly_targets[m] * scale
            if target <= 0:
                continue
            # Final check: do not exceed current cash buffer
            available_now = portfolio.cash_buffer
            purchase_amount = min(target, available_now)
            if purchase_amount <= 0:
                continue
            # approximate coupon/yield at purchase from current_yields mapping
            y = current_yields.get(m, current_yields.get(min(current_yields.keys())))
            portfolio.add_holding(notional=purchase_amount, maturity_months=m, coupon_rate_annual=float(y))
            executed.append({"maturity": m, "requested": monthly_targets[m], "executed": purchase_amount, "yield_at_purchase": y})

        # 6) After attempting all maturities, if some monthly_targets were not executed (because cash ran out),
        #    attempt to invest the shortfall into the shortest tenor (1M) immediately (as per your rule).
        remaining_target = sum(mt * scale for mt in monthly_targets.values()) - sum(e["executed"] for e in executed)
        if remaining_target > 0:
            short_m = 1
            available_now = portfolio.cash_buffer
            if available_now > 0:
                invest_short = min(available_now, remaining_target)
                y_short = current_yields.get(short_m, current_yields.get(min(current_yields.keys())))
                portfolio.add_holding(notional=invest_short, maturity_months=short_m, coupon_rate_annual=float(y_short))
                executed.append({"maturity": short_m, "requested": remaining_target, "executed": invest_short, "yield_at_purchase": y_short})

        # 7) Compose executed summary
        summary = {
            "locked_before": locked_amount,
            "available_by_volume": available_by_volume,
            "cash_buffer_after": portfolio.cash_buffer,
            "executed": executed,
            "outstanding_borrow": portfolio.outstanding_borrow if hasattr(portfolio, "outstanding_borrow") else 0.0
        }

        # 8) Compute simplified margin:
        # margin = sum(bond_volume * yield_decimal) - total_volume * client_rate_decimal
        # sum(bond_volume) uses current holdings (including newly purchased). Use current yields mapping.
        holdings = getattr(portfolio, "holdings", [])
        bond_income = 0.0
        for h in holdings:
            # use current_yields for the holding's remaining tenor if available; fallback to h.yield_at_purchase
            tenor = max(1, h.maturity_months)  # remaining months
            y_current = current_yields.get(tenor, None)
            if y_current is None:
                y_current = getattr(h, "yield_at_purchase", 0.0)
            y_dec = _to_decimal(y_current)
            bond_income += h.notional * y_dec

        client_rate_dec = _to_decimal(client_rate)
        client_cost = total_volume * client_rate_dec

        margin = bond_income - client_cost

        return summary, margin
