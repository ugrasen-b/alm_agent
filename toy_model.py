# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 01:09:03 2025

@author: ugras
"""
import os
import numpy as np
from src.simulation.yield_curve_simulator import YieldCurveSimulator
from src.simulation.client_rate_model import ClientRateModel
from src.simulation.volume_simulator import VolumeSimulator

SEED = 42
np.random.seed(SEED)

T = 12
M = 4
TENORS = [1, 3, 6, 12]
SCEN = 10000

OUT_DIR = "results/scenarios"
os.makedirs(OUT_DIR, exist_ok=True)

# yield-curve model params (percent units)
INIT_CURVE_PCT = np.array([3.0, 3.2, 3.5, 3.8])   # length M
KAPPA = 0.2                                        # mean-reversion speed (scalar)
THETA_PCT = INIT_CURVE_PCT.copy()                  # long-run mean (percent)
SIGMA_PCT = np.array([0.15, 0.12, 0.10, 0.08])     # shock std (percent)

# volume model params (absolute amounts)
BASE_VOL = 1_000_000.0
TREND_PER_MONTH = 2000.0
SEASONALITY_AMPLITUDE = 0.05   # fraction of BASE_VOL (±)
AR_COEF = 0.85
NOISE_STD_FRAC = 0.015        # fraction of base_vol

# client-rate piecewise linear params (percent)
breakpoints = np.array([1.0, 3.0])
slopes = np.array([0.5, 0.8, 1.0])  # len = len(breakpoints)+1
base_intercept = 0.0
lag_months = 1
noise_std_cr = 0.02   # percent

## Yield curve scenarios
sim = YieldCurveSimulator(
    maturities=TENORS,
    horizon=T,
    num_scenarios=SCEN,
    initial_curve=INIT_CURVE_PCT,
    kappa=KAPPA,
    theta=THETA_PCT,
    sigma=SIGMA_PCT,
    seed=SEED
)
yc_paths = sim.simulate()  # shape (S, H, M) in percent

print("Done. Shape:", yc_paths.shape)
print("Example (scenario 0, first 3 months):")
print(yc_paths[0, :3, :])

crm = ClientRateModel(
    maturities=TENORS,
    breakpoints=breakpoints,
    slopes=slopes,
    base_intercept=base_intercept,
    lag=lag_months,
    noise_std=noise_std_cr,
    seed=SEED
)

# compute client rates (returns percent, shape (S, T))
cr = crm.compute(yc_paths)
print("computed cr shape:", cr.shape)
print("sample (scenario 0, months 0..5):", cr[0, :6])

vol_sim = VolumeSimulator(
    horizon=T,
    num_scenarios=SCEN,
    base_volume=BASE_VOL,
    trend_per_month=TREND_PER_MONTH,
    seasonality_amplitude=SEASONALITY_AMPLITUDE,
    ar_coef=AR_COEF,
    noise_std=NOISE_STD_FRAC,
    seed=SEED
)
vol_paths = vol_sim.simulate()   # expected shape (S,T)

print("Done. Shape:", vol_paths.shape)
np.save(os.path.join(OUT_DIR, "vol_paths.npy"), vol_paths)
print("Saved to:", os.path.join(OUT_DIR, "vol_paths.npy"))

BASE_VOL = 1_000_000.0
HIST_MONTHS = 12
TENORS = [1, 3, 6, 12]    # M = 4

# per-month allocation fractions across the 4 tenors (must sum to 1)
FRACTIONS = np.array([0.25, 0.35, 0.20, 0.20], dtype=float)

monthly_total = BASE_VOL / HIST_MONTHS  # per-month notional
# initial_portfolio: each past month has same split; shape (12,4)
initial_portfolio = np.tile(monthly_total * FRACTIONS.reshape(1, -1), (HIST_MONTHS, 1))

# build historical yields (percent units) with a little month-to-month variation
rng = np.random.default_rng(42)
base_yc = np.array([3.0, 3.2, 3.5, 3.8], dtype=float)  # percent
# add a small seasonal wobble and tiny noise
months = np.arange(HIST_MONTHS)
season = 0.05 * np.sin(2 * np.pi * months / 12.0)   # small seasonal shift in percent
historical_yc = np.zeros((HIST_MONTHS, len(TENORS)), dtype=float)
for i in range(HIST_MONTHS):
    noise = rng.normal(loc=0.0, scale=0.02, size=len(TENORS))  # +/- ~0.02% noise
    historical_yc[i] = base_yc + season[i] + noise

def create_maturity_multiplier_matrix(T, M, tenors) -> np.ndarray:
    """
    Create a 12x4 binary matrix indicating active months for each tenor.

    Columns correspond to tenors [1M, 3M, 6M, 12M].
    Each column has 1s for the months in which that bond remains active.

    Returns
    -------
    np.ndarray
        Shape (12, 4), entries are 0 or 1.
    """
    maturities = tenors

    multiplier = np.zeros((T, M), dtype=int)
    for j, m in enumerate(maturities):
        multiplier[-m:, j] = 1  # fill last `m` months with 1s
    return multiplier    

def calculate_margin(
    yc_paths: np.ndarray,
    cr_paths: np.ndarray,
    vol_paths: np.ndarray,
    initial_portfolio: np.ndarray,
    historical_yc: np.ndarray,
    strategy_array: np.ndarray,
) -> np.ndarray:
    """
    Calculate monthly margin across all scenarios.

    Parameters
    ----------
    yc_paths : np.ndarray
        Shape (S, 12, 4) - simulated yield curves in percent.
    cr_paths : np.ndarray
        Shape (S, 12) - simulated client rates in percent.
    vol_paths : np.ndarray
        Shape (S, 12) - total monthly volumes (absolute).
    initial_portfolio : np.ndarray
        Shape (12, 4) - existing active bonds notionals from past months.
    historical_yc : np.ndarray
        Shape (12, 4) - yields (percent) when historical bonds were purchased.
    strategy_array : np.ndarray
        Shape (4,) - allocation fractions for 1M, 3M, 6M, 12M tenors (sum ≈ 1).

    Returns
    -------
    margin_array : np.ndarray
        Shape (S, 12) - margin per scenario per month (decimal or percent).
    """
    # --- sanity checks ---
    S, T, M = yc_paths.shape
    assert cr_paths.shape == (S, T)
    assert vol_paths.shape == (S, T)
    assert initial_portfolio.shape == (T, M)
    assert historical_yc.shape == (T, M)
    assert strategy_array.shape == (M,)

    # --- placeholder output ---
    margin_array = np.zeros((S, T), dtype=float)
    maturity_multiplier = create_maturity_multiplier_matrix(T, M, TENORS)
    active_tranch_multiplier = maturity_multiplier[1:,:]
    tranches_array = np.zeros((S, T+T, M))
    tranches_array[:, :12, :] = initial_portfolio
    
    hist_broadcast = np.broadcast_to(historical_yc[None, :, :], (S, T, M)).copy()
    yc_combined = np.concatenate([hist_broadcast, yc_paths], axis=1)
    
    for t in range(T):
        row_idx = T + t
        vol_t = vol_paths[:,t]
        cr_t = cr_paths[:,t]
        tranches_t = tranches_array[:,:row_idx-1,:]*active_tranch_multiplier
        locked_volume = tranches_t.sum(axis=(1,2))
        free_volume = vol_t - locked_volume
        target_volume = free_volume [:, None]* strategy_array
        target_tranches = target_volume/TENORS
        

    # --- TODO: your implementation here ---
    # You will likely:
    #  1. Combine `initial_portfolio` and new allocations (per strategy)
    #  2. Track active bonds for each tenor and compute income:
    #       bond_income = sum(active_bonds * yield)
    #  3. Compute client_cost = total_volume * client_rate
    #  4. margin = (bond_income - client_cost) / total_volume
    #  5. Repeat per month and per scenario
    #
    # Example outline:
    #
    # for s in range(S):
    #     portfolio = np.zeros((24, M))
    #     purchase_yield = np.zeros((24, M))
    #     portfolio[:12, :] = initial_portfolio
    #     purchase_yield[:12, :] = hist_yc_dec
    #
    #     for t in range(12):
    #         # compute locked, available, add new purchases per strategy
    #         # compute bond_income & client_cost
    #         # margin_array[s, t] = (bond_income - client_cost) / vol_paths[s, t]
    #         pass

    return margin_array

calculate_margin(yc_paths, cr, vol_paths, initial_portfolio, historical_yc, np.array([0.25,0.25,0.25,0.25]))