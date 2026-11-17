import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG (model assumptions)
# -----------------------------
STARTING_CAPITAL = 10_000.0

YEARS = 2
DAYS_PER_YEAR = 252
TRADES_PER_DAY = 4

SIMULATIONS = 50_000

WIN_RATE = 0.40     # probability of a win
R_WIN = 2.0         # +R on wins
R_LOSS = 1.0        # -R on losses

RISK_PER_TRADE = 0.0075   # 0.75% of equity risked each trade
RUIN_FRACTION = 0.50      # ruin if equity < 50% of starting

# -----------------------------
# DERIVED VALUES
# -----------------------------

TOTAL_TRADES = YEARS * DAYS_PER_YEAR * TRADES_PER_DAY
RUIN_LEVEL = STARTING_CAPITAL * RUIN_FRACTION

# -----------------------------
# 1) RANDOM OUTCOMES
# -----------------------------
u = np.random.rand(SIMULATIONS, TOTAL_TRADES)
wins = (u < WIN_RATE)

# -----------------------------
# 2) CONVERT WINS TO R-MULTIPLES
# -----------------------------
r_matrix = np.where(wins, R_WIN, -R_LOSS)

# -----------------------------
# 3) CONVERT R TO PERCENT RETURNS
# -----------------------------
return_matrix = 1 + (RISK_PER_TRADE * r_matrix)

# -----------------------------
# 4) COMPOUND EQUITY PATHS
# -----------------------------
compounded = np.cumprod(return_matrix, axis=1)

equity_paths = STARTING_CAPITAL * compounded

# Add starting capital at time 0
equity_paths = np.column_stack(
    (np.full(SIMULATIONS, STARTING_CAPITAL), equity_paths)
)

# -----------------------------
# 5) COMPUTE RUNNING PEAK
# -----------------------------
running_peak = np.maximum.accumulate(equity_paths, axis=1)

# -----------------------------
# 6) COMPUTE DRAWDOWNS
# -----------------------------
drawdowns = (running_peak - equity_paths) / running_peak

# -----------------------------
# 7) MAX DRAWDOWN PER PATH
# -----------------------------
max_drawdowns = np.max(drawdowns, axis=1)

# -----------------------------
# 8) FINAL EQUITY
# -----------------------------
final_equity = equity_paths[:, -1]

# -----------------------------
# 9) RUIN DETECTION
# -----------------------------
min_equity = np.min(equity_paths, axis=1)
ruin_count = np.sum(min_equity < RUIN_LEVEL)

# -----------------------------
# 10) RISK METRICS
# -----------------------------

mean_final = np.mean(final_equity)
median_final = np.median(final_equity)

var_5 = np.percentile(final_equity, 5)
expected_shortfall_5 = np.mean(final_equity[final_equity <= var_5])

average_max_dd = np.mean(max_drawdowns)
worst_dd = np.max(max_drawdowns)

ruin_probability = ruin_count / SIMULATIONS

# Path-wise percentiles for richer visuals
time_index = np.arange(equity_paths.shape[1])
path_p5, path_median, path_p95 = np.percentile(equity_paths, [5, 50, 95], axis=0)

dd_95 = np.percentile(max_drawdowns, 95)

# -----------------------------
# 11) REPORT
# -----------------------------
print("\n--- Monte Carlo Risk Report ---")
print("Mean Final Equity:", mean_final)
print("Median Final Equity:", median_final)
print("5% VaR (Final Equity):", var_5)
print("Expected Shortfall (5%):", expected_shortfall_5)
print("Average Max Drawdown:", average_max_dd)
print("Worst Drawdown:", worst_dd)
print("95th Percentile Max Drawdown:", dd_95)
print("Probability of Ruin:", ruin_probability)

# -----------------------------
# 12) VISUALIZATION
# -----------------------------

plt.figure(figsize=(18, 5))

# Final equity histogram (log scale)
plt.subplot(1, 3, 1)
log_final = np.log10(final_equity)
plt.hist(log_final, bins=80, color="steelblue", alpha=0.8)
plt.axvline(np.log10(mean_final), color="orange", linestyle="--", label="Mean")
plt.axvline(np.log10(median_final), color="green", linestyle="--", label="Median")
plt.axvline(np.log10(var_5), color="red", linestyle=":", label="5% VaR")
plt.title("Final Equity (log10 scale)")
plt.xlabel("log10(Final Equity)")
plt.ylabel("Frequency")
plt.legend()

# Max drawdown histogram with tail marker
plt.subplot(1, 3, 2)
plt.hist(max_drawdowns, bins=80, color="slategray", alpha=0.8)
plt.axvline(dd_95, color="red", linestyle="--", label="95th percentile")
plt.title("Max Drawdown Distribution")
plt.xlabel("Max Drawdown (fraction of peak)")
plt.ylabel("Frequency")
plt.legend()

# Equity path percentile fan
plt.subplot(1, 3, 3)
plt.fill_between(time_index, path_p5, path_p95, color="lightblue", alpha=0.4, label="5th–95th pct")
plt.plot(time_index, path_median, color="blue", linewidth=1.5, label="Median path")
plt.axhline(RUIN_LEVEL, color="red", linestyle="--", label="Ruin level")
plt.title("Equity Path Percentile Fan")
plt.xlabel("Trade index")
plt.ylabel("Equity")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 13) SHARPE RATIO DISTRIBUTION
# -----------------------------

log_returns = np.log(equity_paths[:, 1:] / equity_paths[:, :-1])

mean_returns = np.mean(log_returns, axis=1)
std_returns = np.std(log_returns, axis=1)

annualization_factor = np.sqrt(DAYS_PER_YEAR * TRADES_PER_DAY)

sharpe_ratios = mean_returns / std_returns * annualization_factor

print("Average Sharpe:", np.mean(sharpe_ratios))
