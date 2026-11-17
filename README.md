## Monte Carlo Risk & Kelly Engine

This repository contains a collection of Monte Carlo experiments for
understanding \("risk of ruin", drawdown, and growth\) for systematic
trading strategies under different sizing rules, including:

- **Fixed-fraction sizing**
- **Half Kelly and full Kelly sizing**
- **Volatility-targeted Kelly sizing**
- **Regime-based volatility-targeted Kelly sizing**

All simulations work in units of **R-multiples** per trade
\(e.g. +2R on a win, -1R on a loss\), and then map those R-multiples
into percentage changes in equity.

### Core scripts

- **`mc_risk_lab.py`**:  
  Baseline Monte Carlo engine for a fixed-fraction strategy. It:
  - Draws Bernoulli win/loss outcomes with configurable win rate and R-multiples  
  - Compounds equity paths over many simulations  
  - Reports final-equity distribution, maximum drawdown, ruin probability, and Sharpe  
  - Produces histograms and sample path plots

- **`mc_risk_lab_kelly.py`**:  
  Compares **fixed-fraction**, **half Kelly**, and **full Kelly** position sizing for
  the same underlying edge. It:
  - Estimates an empirical Kelly fraction \(μ / σ²\) from the simulated R-multiples  
  - Builds equity paths for each sizing rule  
  - Computes and compares final equity, ruin probability, max drawdown, and Sharpe  
  - Visualizes distributions and sample paths side by side

- **`vol_target_kelly.py`**:  
  Explores a **volatility-targeted Kelly** approach where:
  - A global Kelly fraction is estimated from the overall R-distribution  
  - A rolling volatility estimator is used to recompute an effective Kelly fraction
    through time  
  - Leverage is capped at a multiple of full Kelly to avoid extreme bets  
  - Equity paths and risk statistics are computed for this adaptive sizing scheme

- **`src/Regime_vol_target_engine/kelly_engine.py`**:  
  A more advanced **regime-based** volatility-targeted Kelly engine. It:
  - Simulates a Markov regime process with three states \(Calm, Volatile, Crisis\)  
  - Draws R-multiples conditional on regime, plus rare large shock events  
  - Targets a given annualized volatility by scaling trades using rolling volatility  
  - Caps leverage at a fraction of full Kelly  
  - Reports final equity, ruin probability, drawdown, and Sharpe

The other files under `src/Regime_vol_target_engine/` are scaffolding for a more
modular engine \(configuration, regime simulation, and risk-metric helpers\).

### Notebook

- **`Kelly_Monte_Carlo .ipynb`**:  
  A step-by-step, notebook-style walkthrough of the simplest Monte Carlo model:
  - Defines the trading strategy configuration  
  - Builds trade outcome matrices and equity paths  
  - Computes risk metrics like VaR, expected shortfall, and max drawdown  
  - Visualizes distributions and example paths  

Use this notebook as an interactive lab to tweak parameters and build intuition
before moving to the more engineered scripts.

### Running the experiments

All scripts are plain Python and depend primarily on:

- `numpy`
- `matplotlib`

From the project root, you can run for example:

```bash
python src/mc_risk_lab.py
python src/mc_risk_lab_kelly.py
python src/vol_target_kelly.py
python src/Regime_vol_target_engine/kelly_engine.py
```

Each script prints a textual report to the console and, where applicable,
opens matplotlib figures showing the simulated distributions.

