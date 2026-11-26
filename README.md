# cuda deep hedging

A CUDA implementation of a neural network hedger trained with Backpropagation Through Time (BPTT) to hedge an at-the-money European call option under transaction costs, compared against a Black-Scholes delta baseline.

## Why

Black-Scholes delta hedging assumes continuous, frictionless trading. With discrete rebalancing and proportional transaction fees (10 bps), chasing every price move can get expensive near expiry. The goal of this project is to implement the full simulation, forward pass, portfolio accounting, BPTT backward pass, and AdamW optimizer directly in CUDA, verify that the neural hedger recovers Black-Scholes in a frictionless simulated market, and test honestly how it holds up on real S&P 500 data.

## How it works

The model is a small 2-hidden-layer MLP (5 normalized inputs -> 64 -> 64 -> 1 sigmoid delta) evaluated across 29 rebalancing steps (`num_steps = 30`, no trading at expiry). Every step accrues cash at the risk-free rate, updates the hedge position, pays a 10 bps fee on position changes, and unwinds any remaining shares at expiry against the option payoff. Initial option prices use the analytical Black-Scholes formula so day-0 mark-to-market starts cleanly at the opening trade fee.

For real-market testing, `scripts/fetch_data.py` downloads 1-hour S&P 500 (`^GSPC`) bars (`5,007` bars over 720 days, about 7 bars per trading day) and splits the series chronologically before building windows, leaving a 30-bar gap between splits so nothing overlaps: 2,975 training windows, 692 validation windows (used to pick the best checkpoint), and 40 non-overlapping test horizons (`S0 = K = 100` normalized, actual index levels `$6,413–$7,796`). Annualized volatility (`sigma = 0.1623`) is estimated strictly from the training split.

Training minimizes terminal PnL variance plus a `0.25` weight on smooth 95% tail loss (CVaR), using AdamW (`lr = 0.0005`, weight decay `0.05`). Because the tail term is evaluated on uncentered PnL, the network pays for trading fees instead of churning, and its gradients stay invariant to shifts in the starting option price (though on real equity data it is not strictly drift-free).

## Build and run

```bash
# 1. fetch ^GSPC 1-hour data and build train/val/test splits
python scripts/fetch_data.py

# 2. build (Release -O3 by default) and run tests
mkdir -p build && cd build
cmake ..
make -j
ctest --output-on-failure

# 3. train and evaluate on S&P 500
./deep_hedger

# optional: simulated GBM checks (10 bps fee, or 0 bps convergence check)
# ./deep_hedger --sim
# ./deep_hedger --sim --no-cost
# ./deep_hedger --sim --no-cost --lambda 0

# 4. print metrics and regenerate plots
cd .. && python scripts/analysis.py
```

## Results

### Simulated GBM check (`N = 8,192` test paths, `sigma = 0.20`, `T = 29/252` yr)

With 0 bps fees, Black-Scholes is the exact ground truth. Both strategies finish with mean PnL near zero, and the neural policy converges closely to the analytical Black-Scholes delta surface (mean absolute difference `0.0114` for pure variance, `0.0136` with the tail penalty). With 10 bps fees, the policy trades slightly higher variance for lower fee drag and lower overall loss.

| Setting | Strategy | Mean PnL | Std Dev | Variance | Loss | Surface MAE vs BS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 bps (`--no-cost --lambda 0`) | Black-Scholes Delta | -0.0017 | 0.4368 | 0.1908 | 0.1908 | — |
| 0 bps (`--no-cost --lambda 0`) | Deep Hedging Policy | -0.0021 | 0.4507 | 0.2032 | 0.2032 | 0.0114 |
| 0 bps (`--no-cost`) | Black-Scholes Delta | -0.0017 | 0.4368 | 0.1908 | 0.4493 | — |
| 0 bps (`--no-cost`) | Deep Hedging Policy | -0.0013 | 0.4558 | 0.2077 | 0.4564 | 0.0136 |
| 10 bps (`--sim`) | Black-Scholes Delta | -0.2799 | 0.4544 | 0.2065 | 0.5545 | — |
| 10 bps (`--sim`) | Deep Hedging Policy | -0.2659 | 0.4640 | 0.2153 | 0.5345 | — |

### Out-of-sample S&P 500 (`N = 40` non-overlapping horizons, 10 bps fee)

PnL values are in normalized units (`S0 = K = 100`).

| Metric | Black-Scholes Delta | Deep Hedging Policy |
| :--- | :--- | :--- |
| Mean PnL | -0.0595 | -0.0679 |
| Median PnL | -0.0485 | -0.0150 |
| Std Dev | 0.3677 | 0.3971 |
| Variance | 0.1352 | 0.1577 |
| Sharpe Ratio | -0.1617 | -0.1710 |
| Sortino Ratio | -0.1314 | -0.1414 |
| Win Rate (%) | 45.0000 | 47.5000 |
| 95% VaR | -0.5129 | -0.9073 |
| 95% CVaR | -1.1380 | -1.0186 |
| 99% VaR | -1.2729 | -1.0408 |
| 99% CVaR | -1.7513 | -1.1195 |
| Objective Loss | 0.4198 | 0.4125 |
| Mean MSE | 0.1387 | 0.1623 |
| Max Trajectory DD | -0.0626 | -0.0813 |

On the 692 validation windows (`best_epoch = 110`), the neural policy beat Black-Scholes cleanly (std `0.2924` vs `0.3423`, loss `0.2349` vs `0.4012`). On the 40 unseen non-overlapping test horizons, the picture is mixed and heavily shaped by sample size. The policy's better 95%/99% CVaR comes almost entirely from a single selloff horizon (`idx = 38`, a `-2.77%` drop where Black-Scholes lost `-1.7513` and the policy lost `-1.1195`). Without that one horizon, Black-Scholes is ahead on the other 39 horizons (`mean -0.0161` vs `-0.0410`, `std 0.2518` vs `0.3642`, `95% CVaR -0.5185` vs `-0.9121`).

Because the neural policy flattens the sharp delta jump near expiry to save on fees, it under-hedges sharp rallies (`+2.5%` to `+4.0%` horizons), which kept its cumulative 1x-index PnL underwater across the whole test window (`-$173` total, trough `-$192`). Black-Scholes stayed near breakeven and reached `+$53` before dropping on the late selloff (`-$162` total). With only ~3,000 historical 1-hour training bars, a calibrated Black-Scholes baseline remains tough to beat consistently out of sample.

## Plots

### Cumulative equity curve
![Full Equity Curve Comparison](plots/full_equity_curve_comparison.png)

Cumulative PnL scaled by each horizon's actual S&P 500 spot price (`$6,413–$7,796`). The policy stayed negative through the early rallies while Black-Scholes held up better until the late selloff at horizons 38–40.

### Terminal PnL and trajectories
![PnL Trajectory Analysis](plots/pnl_trajectory_analysis.png)

Both strategies start at the opening trade fee (`~ -0.05`) with no artificial day-0 jump. The histogram shows Black-Scholes's tighter core distribution alongside its single `-1.75` outlier.

### Delta surface
![Hedging Surface 3D](plots/hedging_surface_3d.png)

The learned policy follows the Black-Scholes delta shape closely while smoothing out the steep transition near expiry to avoid fee churn.

### Training and validation loss
![Training Losses](plots/training_losses_curve.png)

Training loss keeps falling over 2,000 epochs while validation loss bottoms out around epoch 110, showing why validation checkpointing is needed on limited historical data.
