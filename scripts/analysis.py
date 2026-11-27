import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.optimize import brentq
from scipy.special import ndtr

BG = "#09090B"
GRID = "#26262C"
CHALK = "#EAEAE6"
MUTED = "#8E8E98"
ACCENT = "#5E8BF2"
WARN = "#D97757"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": CHALK,
    "axes.titlecolor": CHALK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": CHALK,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

plots_dir = "plots"
results_dir = "results"
os.makedirs(plots_dir, exist_ok=True)


def save_compact_png(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    # quantize gently if needed to keep all portfolio figures under 250 KB
    if os.path.getsize(path) > 240 * 1024:
        img = Image.open(path).convert("RGB")
        img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def compute_metrics(pnl, name):
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    cvar_95 = np.mean(pnl[pnl <= var_95])
    cvar_99 = np.mean(pnl[pnl <= var_99])
    return {
        "Strategy": name,
        "Mean PnL": np.mean(pnl),
        "Std Dev": np.std(pnl),
        "Variance": np.var(pnl),
        "95% VaR": var_95,
        "95% CVaR": -cvar_95,
        "99% CVaR": -cvar_99,
    }


df_pnl = pd.read_csv(os.path.join(results_dir, "test_pnl_distribution.csv"))
df_meta = pd.read_csv(os.path.join(results_dir, "run_metadata.csv")).iloc[0]
bs_pnl = df_pnl["BS_PnL"].values
dh_pnl = df_pnl["Policy_PnL"].values

df_report = pd.DataFrame([
    compute_metrics(bs_pnl, "BS Delta"),
    compute_metrics(dh_pnl, "Deep Hedging"),
]).set_index("Strategy").T
print(f"OOS Summary ({df_meta['model']}, N={int(df_meta['num_test_paths'])}, {df_meta['cost_ratio']*10000:.0f} bps):")
print(df_report.to_string(float_format=lambda x: f"{x:8.4f}"))

# 1. diverging 2D contour heatmap: Delta_Policy - Delta_BS
surf_path = os.path.join(results_dir, "hedging_surface.csv")
if os.path.exists(surf_path):
    df_surf = pd.read_csv(surf_path)
    moneyness = np.unique(df_surf["Moneyness"].values)
    tau = np.unique(df_surf["Tau_Norm"].values)
    num_s, num_tau = len(moneyness), len(tau)
    pol_grid = df_surf["Policy_Delta"].values.reshape(num_s, num_tau)
    bs_grid = df_surf["BS_Delta"].values.reshape(num_s, num_tau)
    diff_grid = pol_grid - bs_grid

    TAU, MONEYNESS = np.meshgrid(tau, moneyness)
    vmax = max(0.05, float(np.percentile(np.abs(diff_grid), 99)))
    cmap_div = LinearSegmentedColormap.from_list("dark_div", [WARN, "#141418", ACCENT])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cf = ax.contourf(TAU, MONEYNESS, diff_grid, levels=25, cmap=cmap_div, vmin=-vmax, vmax=vmax)
    ax.contour(TAU, MONEYNESS, diff_grid, levels=[-0.05, 0.0, 0.05], colors=CHALK, linewidths=[0.6, 0.9, 0.6], alpha=0.45)
    ax.set_xlabel("tau / T")
    ax.set_ylabel("S / K")
    ax.set_title("Delta_Policy - Delta_BS", loc="left", fontsize=11, pad=10)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED)
    save_compact_png(fig, os.path.join(plots_dir, "delta_diff_surface.png"))

    # 2. borderless dark 3D wireframe mesh (Policy vs BS reference)
    fig = plt.figure(figsize=(7.6, 5.8))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor(BG)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((0.035, 0.035, 0.043, 1.0))
        axis._axinfo["grid"]["color"] = (0.149, 0.149, 0.173, 0.7)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color(GRID)

    stride = max(1, num_s // 24)
    ax3.plot_wireframe(TAU, MONEYNESS, bs_grid, rstride=stride, cstride=stride, color=MUTED, linewidth=0.6, alpha=0.45, label="BS")
    ax3.plot_wireframe(TAU, MONEYNESS, pol_grid, rstride=stride, cstride=stride, color=ACCENT, linewidth=0.9, alpha=0.9, label="Policy")
    ax3.set_xlabel("tau / T", labelpad=8)
    ax3.set_ylabel("S / K", labelpad=8)
    ax3.set_zlabel("Delta", labelpad=6)
    ax3.set_zlim(0.0, 1.0)
    ax3.view_init(elev=24, azim=-128)
    ax3.legend(frameon=False, loc="upper left")
    save_compact_png(fig, os.path.join(plots_dir, "hedging_surface_3d.png"))

# 3. 100k-path OOS PnL density & left-tail CVaR95 overlay
bs_cvar95 = np.mean(bs_pnl[bs_pnl <= np.percentile(bs_pnl, 5)])
dh_cvar95 = np.mean(dh_pnl[dh_pnl <= np.percentile(dh_pnl, 5)])

lo = min(np.percentile(bs_pnl, 0.2), np.percentile(dh_pnl, 0.2))
hi = max(np.percentile(bs_pnl, 99.8), np.percentile(dh_pnl, 99.8))
bins = np.linspace(lo, hi, 110)

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.hist(bs_pnl, bins=bins, density=True, histtype="stepfilled", color=MUTED, alpha=0.22, label="BS Delta")
ax.hist(bs_pnl, bins=bins, density=True, histtype="step", color=MUTED, linewidth=1.2)
ax.hist(dh_pnl, bins=bins, density=True, histtype="stepfilled", color=ACCENT, alpha=0.25, label="Deep Hedging")
ax.hist(dh_pnl, bins=bins, density=True, histtype="step", color=ACCENT, linewidth=1.5)
ax.axvline(bs_cvar95, color=MUTED, linestyle="--", linewidth=1.1, label=f"BS CVaR95 ({bs_cvar95:.2f})")
ax.axvline(dh_cvar95, color=ACCENT, linestyle="--", linewidth=1.3, label=f"Policy CVaR95 ({dh_cvar95:.2f})")
ax.axvline(0.0, color=GRID, linestyle=":", linewidth=1.0)
ax.grid(True, alpha=0.6)
ax.set_xlabel("Terminal PnL (V_T)")
ax.set_ylabel("Density")
ax.set_title("OOS Terminal PnL Distribution", loc="left", fontsize=11, pad=10)
ax.legend(frameon=False, loc="upper left")
save_compact_png(fig, os.path.join(plots_dir, "pnl_tail_distribution.png"))

# 4. friction sweep (0 / 10 / 25 / 50 bps) comparing CVaR95 and turnover
sweep_path = os.path.join(results_dir, "friction_sweep.csv")
if os.path.exists(sweep_path):
    df_sw = pd.read_csv(sweep_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for model_key, dashes in [("gbm", (4, 2.5)), ("heston", ())]:
        sub = df_sw[df_sw["model"] == model_key].sort_values("cost_bps")
        if sub.empty:
            continue
        tag = model_key.upper() if model_key == "gbm" else "Heston"
        style_kw = {"dashes": dashes} if dashes else {}
        ax1.plot(sub["cost_bps"], sub["bs_cvar95"], marker="o", markersize=4, color=MUTED, linewidth=1.3, label=f"{tag} BS", **style_kw)
        ax1.plot(sub["cost_bps"], sub["policy_cvar95"], marker="o", markersize=4, color=ACCENT, linewidth=1.8, label=f"{tag} Policy", **style_kw)

        ax2.plot(sub["cost_bps"], sub["bs_turnover"], marker="o", markersize=4, color=MUTED, linewidth=1.3, **style_kw)
        ax2.plot(sub["cost_bps"], sub["policy_turnover"], marker="o", markersize=4, color=ACCENT, linewidth=1.8, **style_kw)

    ax1.set_xlabel("Cost (bps)")
    ax1.set_ylabel("95% CVaR Loss")
    ax1.set_title("Tail Risk vs Friction", loc="left", fontsize=11, pad=10)
    ax1.grid(True, alpha=0.6)
    ax1.legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=3.0)

    ax2.set_xlabel("Cost (bps)")
    ax2.set_ylabel("Mean Turnover (shares)")
    ax2.set_title("Turnover vs Friction", loc="left", fontsize=11, pad=10)
    ax2.grid(True, alpha=0.6)

    fig.tight_layout()
    save_compact_png(fig, os.path.join(plots_dir, "friction_frontier.png"))


# 5. semi-analytical Heston implied volatility surface (Albrecher et al. Little Heston Trap)
def heston_cf(u, T, r, v0, theta, kappa, xi, rho):
    b = kappa - 1j * rho * xi * u
    d = np.sqrt(b * b + (xi * xi) * (1j * u + u * u))
    d = np.where(np.real(d) < 0.0, -d, d)
    g = (b - d) / (b + d)
    exp_neg_dt = np.exp(-d * T)
    C = 1j * u * r * T + (kappa * theta / (xi * xi)) * (
        (b - d) * T - 2.0 * np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    )
    D = ((b - d) / (xi * xi)) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt))
    return np.exp(C + D * v0)


def build_quad_nodes():
    # multi-interval Gauss-Legendre up to u=2000 for smooth wing convergence
    edges = [0.0, 20.0, 100.0, 500.0, 2000.0]
    us, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = np.polynomial.legendre.leggauss(128)
        us.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(us), np.concatenate(ws)


U_QUAD, W_QUAD = build_quad_nodes()


def heston_otm_price(S0, K_arr, r, T, v0, theta, kappa, xi, rho):
    # Lewis (2001) single-integral with BS control variate avoids 0.5+int and S0-K cancellation
    K_arr = np.atleast_1d(np.asarray(K_arr, dtype=np.float64))
    k = np.log(S0 / K_arr) + r * T
    z = U_QUAD - 0.5j
    phi_h = heston_cf(z, T, 0.0, v0, theta, kappa, xi, rho)[None, :]
    phi_bs = np.exp(-0.5 * v0 * T * (z * z + 1j * z))[None, :]
    integrand = np.real(np.exp(1j * U_QUAD[None, :] * k[:, None]) * (phi_h - phi_bs)) / (U_QUAD[None, :] ** 2 + 0.25)
    diff = -np.sqrt(S0 * K_arr) * np.exp(-0.5 * r * T) * (integrand @ W_QUAD) / np.pi

    rt_T = np.sqrt(v0 * T)
    d1 = (k + 0.5 * v0 * T) / rt_T
    d2 = d1 - rt_T
    disc_K = K_arr * np.exp(-r * T)
    bs_otm = np.where(K_arr < S0, disc_K * ndtr(-d2) - S0 * ndtr(-d1), S0 * ndtr(d1) - disc_K * ndtr(d2))
    return bs_otm + diff


def bs_otm_scalar(S0, K, r, T, sigma):
    rt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / rt_T
    d2 = d1 - rt_T
    disc_K = K * np.exp(-r * T)
    if K < S0:
        return disc_K * ndtr(-d2) - S0 * ndtr(-d1)
    return S0 * ndtr(d1) - disc_K * ndtr(d2)


def implied_vol_from_otm(otm_price, S0, K, r, T):
    return brentq(lambda sig: bs_otm_scalar(S0, K, r, T, sig) - otm_price, 1e-4, 3.0, xtol=1e-9)


S0_H, R_H = 100.0, 0.05
V0_H, THETA_H, KAPPA_H, XI_H, RHO_H = 0.04, 0.04, 2.0, 0.50, -0.70

c_atm_30d = float(heston_otm_price(S0_H, [100.0], R_H, 30.0 / 252.0, V0_H, THETA_H, KAPPA_H, XI_H, RHO_H)[0])
mc_ref = float(df_meta["option_price"]) if str(df_meta["model"]).lower() == "heston" else 3.01858
abs_diff_atm = abs(c_atm_30d - mc_ref)
print(f"Heston 30d ATM Call check: semi-analytical={c_atm_30d:.4f}, GPU_MC={mc_ref:.4f}, abs_diff={abs_diff_atm:.4f}")
assert abs_diff_atm < 0.02, f"Heston ATM check failed: {c_atm_30d:.4f} vs {mc_ref:.4f}"

m_grid = np.linspace(0.80, 1.20, 35)
T_grid = np.linspace(7.0 / 252.0, 90.0 / 252.0, 25)
K_grid = S0_H * m_grid

iv_surface = np.zeros((len(T_grid), len(m_grid)), dtype=np.float64)
for i, T_val in enumerate(T_grid):
    otm_prices = heston_otm_price(S0_H, K_grid, R_H, float(T_val), V0_H, THETA_H, KAPPA_H, XI_H, RHO_H)
    for j, p_val in enumerate(otm_prices):
        iv_surface[i, j] = 100.0 * implied_vol_from_otm(float(p_val), S0_H, float(K_grid[j]), R_H, float(T_val))

atm_30d_iv = 100.0 * implied_vol_from_otm(c_atm_30d, S0_H, 100.0, R_H, 30.0 / 252.0)
print(f"Heston IV surface range: [{iv_surface.min():.2f}%, {iv_surface.max():.2f}%], 30d ATM IV={atm_30d_iv:.2f}%")

M_MESH, DAYS_MESH = np.meshgrid(m_grid, T_grid * 252.0)
cmap_iv = LinearSegmentedColormap.from_list("dark_iv", [ACCENT, "#3E4A6B", WARN])

fig = plt.figure(figsize=(7.6, 5.8))
ax_iv = fig.add_subplot(111, projection="3d")
ax_iv.set_facecolor(BG)
for axis in (ax_iv.xaxis, ax_iv.yaxis, ax_iv.zaxis):
    axis.set_pane_color((0.035, 0.035, 0.043, 1.0))
    axis._axinfo["grid"]["color"] = (0.149, 0.149, 0.173, 0.7)
    axis._axinfo["grid"]["linewidth"] = 0.5
    axis.line.set_color(GRID)

ax_iv.plot_surface(
    M_MESH, DAYS_MESH, iv_surface,
    cmap=cmap_iv, edgecolor=BG, linewidth=0.35, alpha=0.92, antialiased=True
)
ax_iv.set_xlabel("K / S0", labelpad=8)
ax_iv.set_ylabel("T (days)", labelpad=8)
ax_iv.set_zlabel("IV (%)", labelpad=6)
ax_iv.view_init(elev=25, azim=42)
save_compact_png(fig, os.path.join(plots_dir, "heston_iv_surface.png"))

print("saved dark-editorial plots to plots/")
