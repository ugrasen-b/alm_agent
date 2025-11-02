# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 00:13:59 2025

@author: ugras
"""

# src/utils/config.py
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "experiment": {"name": "run", "seed": 42, "results_dir": "results/run"},
    "simulation": {"horizon_months": 12, "historical_months": 12, "num_scenarios": 10000},
    "tenors": [1, 3, 6, 12],
    "yield_curve": {},
    "client_rate": {},
    "volume": {},
    "strategy": {"default_fraction_per_tenor": [0.0, 0.6, 0.0, 0.4], "executor": "standard_rollover"},
    "portfolio": {"initial_portfolio_path": None, "historical_yc_path": None, "locked_as_fraction_of_volume": 0.9},
    "run": {"vectorized_scenarios": False, "save_logs": True}
}

def load_config(path="C:/codebase/alm_agent/config.yaml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as f:
        cfg_user = yaml.safe_load(f)
    # shallow merge (user keys override defaults)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(cfg_user or {})
    # basic validation / normalization
    # ensure tenors are ints
    cfg["tenors"] = [int(x) for x in cfg.get("tenors", DEFAULT_CONFIG["tenors"])]
    # ensure horizon/historical sizes
    sim = cfg["simulation"]
    sim["horizon_months"] = int(sim.get("horizon_months", 12))
    sim["historical_months"] = int(sim.get("historical_months", 12))
    sim["num_scenarios"] = int(sim.get("num_scenarios", 10000))
    # ensure strategy vector length matches tenors (if present)
    strat = cfg.get("strategy", {})
    default_frac = strat.get("default_fraction_per_tenor", [])
    if default_frac and len(default_frac) != len(cfg["tenors"]):
        raise ValueError("strategy.default_fraction_per_tenor length must match tenors length")
    return cfg
