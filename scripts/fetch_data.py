import os
import yfinance as yf
import numpy as np


def build_windows(price_slice, window_size, stride):
    paths = []
    payoffs = []
    strikes = []

    for t in range(0, len(price_slice) - window_size + 1, stride):
        window = price_slice[t : t + window_size]
        s0 = window[0]
        st = window[-1]

        k = s0
        payoff = max(st - k, 0.0)

        paths.append(window)
        payoffs.append(payoff)
        strikes.append(k)

    return np.array(paths), np.array(payoffs), np.array(strikes)


def generate_sp500_dataset(ticker="^GSPC", window_size=30, trading_bars_per_day=7.0):
    os.makedirs("data", exist_ok=True)

    print(f"fetching 1-hour historical data for {ticker} (S&P 500)...")
    df = yf.download(ticker, period="720d", interval="1h", progress=False)

    if df.empty:
        raise RuntimeError(f"failed to download price data for {ticker}")

    close_series = df["Close"].dropna()
    prices = close_series.values.flatten()
    num_candles = len(prices)

    if num_candles < 4 * window_size:
        raise RuntimeError(f"insufficient candle count: got {num_candles}, need at least {4 * window_size}")

    # split price series chronologically before windowing with >= window_size purge
    purge_bars = window_size
    train_end = int(num_candles * 0.60)
    val_start = train_end + purge_bars
    val_end = int(num_candles * 0.75)
    test_start = val_end + purge_bars

    if val_start + window_size > val_end or test_start + window_size > num_candles:
        raise RuntimeError("insufficient data after chronological split and purge gaps")

    train_prices = prices[:train_end]
    val_prices = prices[val_start:val_end]
    test_prices = prices[test_start:]

    # stride=1 for train/val windows; stride=window_size for independent non-overlapping test horizons
    train_paths, train_payoffs, train_strikes = build_windows(train_prices, window_size, stride=1)
    val_paths, val_payoffs, val_strikes = build_windows(val_prices, window_size, stride=1)
    test_paths, test_payoffs, test_strikes = build_windows(test_prices, window_size, stride=window_size)

    # estimate annualized volatility strictly on training split
    train_log_ret = np.diff(np.log(train_prices))
    bars_per_year = 252.0 * trading_bars_per_day
    train_sigma = float(np.std(train_log_ret, ddof=1) * np.sqrt(bars_per_year))
    obs_bars_per_day = float(close_series.groupby(close_series.index.date).size().mean())

    print(f"downloaded {num_candles} 1h candles (observed mean {obs_bars_per_day:.2f} bars/trading day).")
    print(f"chronological split (purge={purge_bars} bars): "
          f"train={len(train_prices)} bars -> {len(train_paths)} windows, "
          f"val={len(val_prices)} bars -> {len(val_paths)} windows, "
          f"test={len(test_prices)} bars -> {len(test_paths)} non-overlapping horizons.")
    print(f"train-calibrated annualized sigma (252 * {trading_bars_per_day:.0f}): {train_sigma:.4f}")

    save_dataset("data/GSPC_train_paths.csv", train_paths, train_payoffs, train_strikes)
    save_dataset("data/GSPC_val_paths.csv", val_paths, val_payoffs, val_strikes)
    save_dataset("data/GSPC_test_paths.csv", test_paths, test_payoffs, test_strikes)


def save_dataset(filename, paths, payoffs, strikes):
    if len(paths) == 0:
        print(f"skipping empty dataset for {filename}")
        return

    num_samples, num_steps = paths.shape
    with open(filename, "w") as f:
        f.write(f"{num_samples},{num_steps}\n")
        for i in range(num_samples):
            path_str = ",".join([f"{val:.4f}" for val in paths[i]])
            f.write(f"{strikes[i]:.4f},{payoffs[i]:.4f},{path_str}\n")
    print(f"saved {filename} successfully.")


if __name__ == "__main__":
    generate_sp500_dataset()
