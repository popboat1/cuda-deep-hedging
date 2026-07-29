import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

search_dir = "results" if os.path.exists("results/test_pnl_distribution.csv") else "."
data_dir = "data" if os.path.exists("data/GSPC_test_paths.csv") else "."

pnl_file = os.path.join(search_dir, "test_pnl_distribution.csv")
traj_file = os.path.join(search_dir, "test_equity_trajectories.csv")
sample_file = os.path.join(search_dir, "sample_equity_paths.csv")
surface_file = os.path.join(search_dir, "hedging_surface.csv")
test_paths_file = os.path.join(data_dir, "GSPC_test_paths.csv")

print(f"loading csv data from: {search_dir}")
print(f"saving plots to: {plots_dir}")

df_pnl = pd.read_csv(pnl_file)
df_traj = pd.read_csv(traj_file)
df_sample = pd.read_csv(sample_file) if os.path.exists(sample_file) else None

bs_pnl = df_pnl['BS_PnL'].values
dh_pnl = df_pnl['Policy_PnL'].values

sp500_spot_prices = []
if os.path.exists(test_paths_file):
    with open(test_paths_file, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            if line.strip():
                parts = line.strip().split(",")
                s0 = float(parts[2])
                sp500_spot_prices.append(s0)

sp500_spot_prices = np.array(sp500_spot_prices)

def compute_metrics(pnl, name):
    mean_pnl = np.mean(pnl)
    median_pnl = np.median(pnl)
    std_pnl = np.std(pnl)
    
    downside_returns = pnl[pnl < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-6
    
    sharpe = mean_pnl / (std_pnl + 1e-6)
    sortino = mean_pnl / (downside_std + 1e-6)
    win_rate = np.mean(pnl > 0) * 100.0
    
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    cvar_95 = np.mean(pnl[pnl <= var_95])
    cvar_99 = np.mean(pnl[pnl <= var_99])
    
    mse = np.mean(pnl**2)
    
    return {
        "Strategy": name,
        "Mean PnL": mean_pnl,
        "Median PnL": median_pnl,
        "Std Dev": std_pnl,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Win Rate (%)": win_rate,
        "95% VaR": var_95,
        "95% CVaR (Expected Shortfall)": cvar_95,
        "99% VaR": var_99,
        "99% CVaR": cvar_99,
        "Mean MSE": mse
    }

def calculate_max_dd(series):
    cum_max = np.maximum.accumulate(series)
    drawdowns = series - cum_max
    return np.min(drawdowns)

bs_metrics = compute_metrics(bs_pnl, "Black-Scholes Delta")
dh_metrics = compute_metrics(dh_pnl, "Deep Hedging Policy")

bs_metrics["Max Trajectory DD"] = calculate_max_dd(df_traj['bs_mean'].values)
dh_metrics["Max Trajectory DD"] = calculate_max_dd(df_traj['policy_mean'].values)

df_report = pd.DataFrame([bs_metrics, dh_metrics]).set_index("Strategy").T

print("Performance Comparison (S&P 500):")
print(df_report.to_string(float_format=lambda x: f"{x:8.4f}"))

contract_notional = 5500.0 # S&P 500 spot benchmark
scale_factor = contract_notional / 100.0

bs_pnl_usd = bs_pnl * scale_factor
policy_pnl_usd = dh_pnl * scale_factor

cum_bs = np.cumsum(bs_pnl_usd)
cum_policy = np.cumsum(policy_pnl_usd)

highwater_bs = np.maximum.accumulate(cum_bs)
dd_bs = cum_bs - highwater_bs

highwater_policy = np.maximum.accumulate(cum_policy)
dd_policy = cum_policy - highwater_policy

max_dd_bs = np.min(dd_bs)
max_dd_policy = np.min(dd_policy)
total_trades = len(df_pnl)

fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1.2, 2, 1]})
trade_index = np.arange(1, total_trades + 1)

if len(sp500_spot_prices) == total_trades:
    ax0.plot(trade_index, sp500_spot_prices, color='forestgreen', linewidth=1.5, label='S&P 500 Index ($GSPC)')
    ax0.set_title("S&P 500 Spot Price Trajectory", fontsize=11, fontweight='bold')
    ax0.set_ylabel("Index Level ($)", fontsize=10)
    ax0.legend(loc='upper left', fontsize=9)
    ax0.grid(True, alpha=0.3)

ax1.plot(trade_index, cum_bs, label=f'Black-Scholes Delta (Total: ${cum_bs[-1]:,.0f})', color='crimson', linewidth=1.8)
ax1.plot(trade_index, cum_policy, label=f'Deep Hedging Policy (Total: ${cum_policy[-1]:,.0f})', color='royalblue', linewidth=2.0)
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.set_title("Full Test Set Cumulative Equity Curve (S&P 500 @ $5,500 Spot)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Cumulative Realized PnL ($)", fontsize=10)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.fill_between(trade_index, dd_bs, 0, color='crimson', alpha=0.3, label=f'BS Max DD: ${max_dd_bs:,.0f}')
ax2.fill_between(trade_index, dd_policy, 0, color='royalblue', alpha=0.3, label=f'Deep Hedging Max DD: ${max_dd_policy:,.0f}')
ax2.set_title("Underwater Drawdown ($)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Sequential Test Trade Index", fontsize=10)
ax2.set_ylabel("Drawdown ($)", fontsize=10)
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(plots_dir, "full_equity_curve_comparison.png")
plt.savefig(plot1_path, dpi=300)
plt.close()

# PnL distribution histogram and trajectory
fig2, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].hist(bs_pnl, bins=150, alpha=0.5, label='Black-Scholes Delta', color='crimson', density=True)
axes[0].hist(dh_pnl, bins=150, alpha=0.6, label='Deep Hedging Policy', color='royalblue', density=True)
axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title("Terminal Hedging Error Distribution ($V_T$)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Terminal Hedging Error ($V_T$)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

steps = df_traj['step'].values
axes[1].plot(steps, df_traj['bs_mean'], label='Black-Scholes Mean $V_t$', color='crimson', linewidth=2)
axes[1].fill_between(steps, df_traj['bs_mean'] - df_traj['bs_std'], df_traj['bs_mean'] + df_traj['bs_std'], color='crimson', alpha=0.15)

axes[1].plot(steps, df_traj['policy_mean'], label='Deep Hedging Mean $V_t$', color='royalblue', linewidth=2)
axes[1].fill_between(steps, df_traj['policy_mean'] - df_traj['policy_std'], df_traj['policy_mean'] + df_traj['policy_std'], color='royalblue', alpha=0.15)

axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title(r"Mean Net Equity Trajectory ($\pm 1\sigma$ Band)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time Step $t$ (0 to 29)")
axes[1].set_ylabel("Net Hedging Error ($V_t$)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

if df_sample is not None:
    sample_paths = df_sample['path_idx'].unique()[:5]
    for p_idx in sample_paths:
        path_data = df_sample[df_sample['path_idx'] == p_idx]
        axes[2].plot(path_data['step'], path_data['bs_pnl'], color='crimson', alpha=0.4, linestyle='--')
        axes[2].plot(path_data['step'], path_data['policy_pnl'], color='royalblue', alpha=0.6)

axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[2].set_title("Sample Path Net Trajectories (5 Paths)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Time Step $t$ (0 to 29)")
axes[2].set_ylabel("Net Hedging Error ($V_t$)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(plots_dir, "pnl_trajectory_analysis.png")
plt.savefig(plot2_path, dpi=300)
plt.close()

# 3D hedging surface
if os.path.exists(surface_file):
    df_surf = pd.read_csv(surface_file)

    moneyness = np.unique(df_surf['Moneyness'].values)
    tau = np.unique(df_surf['Tau_Norm'].values)

    num_s = len(moneyness)
    num_tau = len(tau)

    policy_surface = df_surf['Policy_Delta'].values.reshape(num_s, num_tau)
    bs_surface = df_surf['BS_Delta'].values.reshape(num_s, num_tau)

    TAU, MONEYNESS = np.meshgrid(tau, moneyness)

    fig3 = plt.figure(figsize=(20, 9))

    ax3d_1 = fig3.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax3d_1.plot_surface(TAU, MONEYNESS, bs_surface, cmap='viridis', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    ax3d_1.set_title('Theoretical Black-Scholes Delta Surface', fontsize=14, fontweight='bold', pad=15)
    ax3d_1.set_xlabel('Time to Expiry (tau / T)', fontsize=11, labelpad=10)
    ax3d_1.set_ylabel('Moneyness (S / K)', fontsize=11, labelpad=10)
    ax3d_1.set_zlabel('Hedge Ratio (Delta)', fontsize=11, labelpad=10)
    ax3d_1.view_init(elev=25, azim=-125)
    fig3.colorbar(surf1, ax=ax3d_1, shrink=0.5, aspect=10, label='BS Delta')

    ax3d_2 = fig3.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax3d_2.plot_surface(TAU, MONEYNESS, policy_surface, cmap='plasma', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    ax3d_2.set_title('Deep Hedging Neural Policy Delta Surface', fontsize=14, fontweight='bold', pad=15)
    ax3d_2.set_xlabel('Time to Expiry (tau / T)', fontsize=11, labelpad=10)
    ax3d_2.set_ylabel('Moneyness (S / K)', fontsize=11, labelpad=10)
    ax3d_2.set_zlabel('Hedge Ratio (Delta)', fontsize=11, labelpad=10)
    ax3d_2.view_init(elev=25, azim=-125)
    fig3.colorbar(surf2, ax=ax3d_2, shrink=0.5, aspect=10, label='Policy Delta')

    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "hedging_surface_3d.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()

# Training and validation losses plot
losses_file = os.path.join(search_dir, "training_losses.csv")
if os.path.exists(losses_file):
    df_losses = pd.read_csv(losses_file)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    epochs = df_losses['epoch']
    
    ax4.plot(epochs, df_losses['train_hybrid'], label='Hybrid Loss', color='royalblue', linewidth=1.5)
    ax4.plot(epochs, df_losses['train_mse'], label='MSE', color='cornflowerblue', linewidth=1.2, linestyle='--')
    ax4.plot(epochs, df_losses['train_cvar'], label='CVaR', color='darkblue', linewidth=1.2, linestyle=':')
    
    ax4.set_title("Training Losses over Epochs", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Loss", fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plot4_path = os.path.join(plots_dir, "training_losses_curve.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()

print("analysis complete: all plots saved to plots/")