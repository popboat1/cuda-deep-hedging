import os
import yfinance as yf
import numpy as np

def generate_sp500_dataset(ticker="^GSPC", window_size=30):
    os.makedirs("data", exist_ok=True)
    
    print(f"fetching 1-hour historical data for {ticker} (S&P 500)...")
    df = yf.download(ticker, period="720d", interval="1h", progress=False)
    
    if df.empty:
        raise RuntimeError(f"failed to download price data for {ticker}")
        
    close_series = df['Close'].dropna()
    prices = close_series.values.flatten()
    num_candles = len(prices)
    
    if num_candles < window_size:
        raise RuntimeError(f"insufficient candle count: got {num_candles}, need at least {window_size}")
    
    paths = []
    payoffs = []
    strikes = []
    
    stride = 1
    for t in range(0, num_candles - window_size, stride):
        window = prices[t : t + window_size]
        s0 = window[0]
        st = window[-1]
        
        k = s0
        payoff = max(st - k, 0.0)
        
        paths.append(window)
        payoffs.append(payoff)
        strikes.append(k)
        
    paths = np.array(paths)
    payoffs = np.array(payoffs)
    strikes = np.array(strikes)
    
    num_samples = len(paths)
    print(f"generated {num_samples} historical S&P 500 rolling 30-step paths.")
    
    split_idx = int(num_samples * 0.8)
    
    train_paths, test_paths = paths[:split_idx], paths[split_idx:]
    train_payoffs, test_payoffs = payoffs[:split_idx], payoffs[split_idx:]
    train_strikes, test_strikes = strikes[:split_idx], strikes[split_idx:]
    
    save_dataset("data/GSPC_train_paths.csv", train_paths, train_payoffs, train_strikes)
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