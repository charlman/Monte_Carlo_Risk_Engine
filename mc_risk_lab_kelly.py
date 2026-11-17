"""
Monte Carlo Kelly Comparison Engine
------------------------------------

Purpose:
Simulate a multi-year trading strategy and compare
three capital allocation methods:

1) Fixed fractional risk
2) Half Kelly
3) Full Kelly

This engine:
- Generates trade outcomes
- Estimates empirical Kelly fraction
- Compounds equity paths
- Computes risk metrics
- Compares growth vs drawdown vs ruin probability

Author: Charles Dickson
Version: v1.6
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# CONFIGURATION
# =====================================================

STARTING_CAPITAL = 10_000.0

YEARS = 2
DAYS_PER_YEAR = 252
TRADES_PER_DAY = 4

SIMULATIONS = 50_000

WIN_RATE = 0.40
R_WIN = 2.0
R_LOSS = 1.0

FIXED_FRACTION = 0.0075
RUIN_FRACTION = 0.50


# =====================================================
# DERIVED VALUES
# =====================================================

TOTAL_TRADES = YEARS * DAYS_PER_YEAR * TRADES_PER_DAY
RUIN_LEVEL = STARTING_CAPITAL * RUIN_FRACTION


# =====================================================
# 1) GENERATE RANDOM TRADE OUTCOMES
# =====================================================

# Uniform random matrix
u = np.random.rand(SIMULATIONS, TOTAL_TRADES)

# Convert to win/loss boolean matrix
wins = (u < WIN_RATE)

# Convert to R-multiples
r_matrix = np.where(wins, R_WIN, -R_LOSS)


# =====================================================
# 2) ESTIMATE EMPIRICAL KELLY FRACTION
# =====================================================

"""
We treat R-multiples as per-trade returns.
Empirical Kelly formula (continuous approximation):

    f* = μ / σ²

Where:
    μ = mean return per trade
    σ² = variance of return per trade
"""

mu = np.mean(r_matrix)
sigma_squared = np.var(r_matrix)

kelly_fraction = mu / sigma_squared

half_kelly = 0.5 * kelly_fraction
full_kelly = kelly_fraction

print("Empirical Kelly Fraction:", kelly_fraction)


# =====================================================
# 3) BUILD RETURN MATRICES FOR EACH SIZING RULE
# =====================================================

"""
Convert R-multiples into growth multipliers:

    return = 1 + (fraction * R)
"""

return_fixed = 1 + (FIXED_FRACTION * r_matrix)
return_half = 1 + (half_kelly * r_matrix)
return_full = 1 + (full_kelly * r_matrix)


# =====================================================
# 4) EQUITY COMPOUNDING FUNCTION
# =====================================================

def compound_paths(return_matrix):
    """
    Compounds trade multipliers into full equity paths.

    Parameters
    ----------
    return_matrix : np.ndarray
        Matrix of per-trade multipliers
        Shape: (simulations, trades)

    Returns
    -------
    equity_matrix : np.ndarray
        Full equity paths including time 0
        Shape: (simulations, trades + 1)
    """

    compounded = np.cumprod(return_matrix, axis=1)

    equity = STARTING_CAPITAL * compounded

    equity = np.column_stack(
        (np.full(SIMULATIONS, STARTING_CAPITAL), equity)
    )

    return equity


equity_fixed = compound_paths(return_fixed)
equity_half = compound_paths(return_half)
equity_full = compound_paths(return_full)


# =====================================================
# 5) RISK METRIC FUNCTIONS
# =====================================================

def compute_max_drawdown(equity_matrix):
    """
    Computes maximum drawdown for each simulation path.
    """
    running_peak = np.maximum.accumulate(equity_matrix, axis=1)
    drawdowns = (running_peak - equity_matrix) / running_peak
    return np.max(drawdowns, axis=1)


def compute_ruin_probability(equity_matrix):
    """
    Computes probability of hitting ruin threshold.
    """
    min_equity = np.min(equity_matrix, axis=1)
    return np.mean(min_equity < RUIN_LEVEL)


def compute_sharpe(equity_matrix):
    """
    Computes annualized Sharpe ratio per simulation.
    """
    log_returns = np.log(equity_matrix[:, 1:] / equity_matrix[:, :-1])
    mean_returns = np.mean(log_returns, axis=1)
    std_returns = np.std(log_returns, axis=1)

    annualization_factor = np.sqrt(DAYS_PER_YEAR * TRADES_PER_DAY)

    return mean_returns / std_returns * annualization_factor


# =====================================================
# 6) COMPUTE METRICS FOR EACH STRATEGY
# =====================================================

final_fixed = equity_fixed[:, -1]
final_half = equity_half[:, -1]
final_full = equity_full[:, -1]

max_dd_fixed = compute_max_drawdown(equity_fixed)
max_dd_half = compute_max_drawdown(equity_half)
max_dd_full = compute_max_drawdown(equity_full)

ruin_fixed = compute_ruin_probability(equity_fixed)
ruin_half = compute_ruin_probability(equity_half)
ruin_full = compute_ruin_probability(equity_full)

sharpe_fixed = compute_sharpe(equity_fixed)
sharpe_half = compute_sharpe(equity_half)
sharpe_full = compute_sharpe(equity_full)


# =====================================================
# 7) REPORT
# =====================================================

print("\n--- Kelly Strategy Comparison ---")

print("\nMean Final Equity:")
print("Fixed:", np.mean(final_fixed))
print("Half Kelly:", np.mean(final_half))
print("Full Kelly:", np.mean(final_full))

print("\nRuin Probability:")
print("Fixed:", ruin_fixed)
print("Half Kelly:", ruin_half)
print("Full Kelly:", ruin_full)

print("\nAverage Max Drawdown:")
print("Fixed:", np.mean(max_dd_fixed))
print("Half Kelly:", np.mean(max_dd_half))
print("Full Kelly:", np.mean(max_dd_full))

print("\nAverage Sharpe:")
print("Fixed:", np.mean(sharpe_fixed))
print("Half Kelly:", np.mean(sharpe_half))
print("Full Kelly:", np.mean(sharpe_full))

# =====================================================
# VISUALIZATION — RISK / RETURN OVERVIEW
# =====================================================

plt.figure(figsize=(20, 10))

# -----------------------------------------------------
# ROW 1 — DISTRIBUTIONS
# -----------------------------------------------------

# 1) Log Final Equity
plt.subplot(2, 3, 1)
plt.hist(np.log10(final_fixed), bins=100, alpha=0.6, label="Fixed")
plt.hist(np.log10(final_half), bins=100, alpha=0.6, label="Half Kelly")
plt.hist(np.log10(final_full), bins=100, alpha=0.6, label="Full Kelly")
plt.title("Log10 Final Equity")
plt.xlabel("log10(Final Equity)")
plt.ylabel("Frequency")
plt.legend()

# 2) Max Drawdown
plt.subplot(2, 3, 2)
plt.hist(max_dd_fixed, bins=100, alpha=0.6, label="Fixed")
plt.hist(max_dd_half, bins=100, alpha=0.6, label="Half Kelly")
plt.hist(max_dd_full, bins=100, alpha=0.6, label="Full Kelly")
plt.title("Max Drawdown")
plt.xlabel("Max Drawdown (fraction of peak)")
plt.ylabel("Frequency")
plt.legend()

# 3) Sharpe
plt.subplot(2, 3, 3)
plt.hist(sharpe_fixed, bins=100, alpha=0.6, label="Fixed")
plt.hist(sharpe_half, bins=100, alpha=0.6, label="Half Kelly")
plt.hist(sharpe_full, bins=100, alpha=0.6, label="Full Kelly")
plt.title("Sharpe Distribution")
plt.xlabel("Sharpe ratio")
plt.ylabel("Frequency")
plt.legend()

# -----------------------------------------------------
# ROW 2 — RISK VS RETURN SCATTER
# -----------------------------------------------------

n_sample = min(5000, SIMULATIONS)
idx_sample = np.random.choice(SIMULATIONS, size=n_sample, replace=False)

log_final_fixed = np.log10(final_fixed[idx_sample])
log_final_half = np.log10(final_half[idx_sample])
log_final_full = np.log10(final_full[idx_sample])

# Fixed: max drawdown vs final equity
plt.subplot(2, 3, 4)
plt.scatter(max_dd_fixed[idx_sample], log_final_fixed, s=5, alpha=0.25, label="Fixed")
plt.title("Fixed: Risk vs Return")
plt.xlabel("Max Drawdown (fraction of peak)")
plt.ylabel("log10(Final Equity)")
plt.legend()

# Half Kelly: max drawdown vs final equity
plt.subplot(2, 3, 5)
plt.scatter(max_dd_half[idx_sample], log_final_half, s=5, alpha=0.25, label="Half Kelly", color="orange")
plt.title("Half Kelly: Risk vs Return")
plt.xlabel("Max Drawdown (fraction of peak)")
plt.ylabel("log10(Final Equity)")
plt.legend()

# Full Kelly: max drawdown vs final equity
plt.subplot(2, 3, 6)
plt.scatter(max_dd_full[idx_sample], log_final_full, s=5, alpha=0.25, label="Full Kelly", color="red")
plt.title("Full Kelly: Risk vs Return")
plt.xlabel("Max Drawdown (fraction of peak)")
plt.ylabel("log10(Final Equity)")
plt.legend()

plt.tight_layout()
plt.show()
