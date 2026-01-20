# ==========================================================
# IMPORTS
# ==========================================================

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

# ==========================================================
# PARAMETERS
# ==========================================================

end_date = dt.date.today()
start_date = end_date - pd.Timedelta(days=5 * 252)

nifty_top_20 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "NTPC.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS"
]

MAX_MA = 80

final_results = []

# ==========================================================
# LOOP OVER STOCKS
# ==========================================================

for ticker in nifty_top_20:

    print(f"Processing {ticker}")

    # -----------------------------
    # DOWNLOAD DATA
    # -----------------------------
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        continue

    close = df["Close"].dropna().values
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan

    # -----------------------------
    # PRECOMPUTE MOVING AVERAGES
    # -----------------------------
    ma_matrix = np.zeros((len(close), MAX_MA + 1))

    for ma in range(1, MAX_MA + 1):
        ma_matrix[:, ma] = pd.Series(close).rolling(ma).mean().values

    ma_prev_matrix = np.roll(ma_matrix, 1, axis=0)
    ma_prev_matrix[0, :] = np.nan

    # Valid rows
    valid = ~np.isnan(ma_matrix).any(axis=1)
    close = close[valid]
    close_prev = close_prev[valid]
    ma_matrix = ma_matrix[valid]
    ma_prev_matrix = ma_prev_matrix[valid]

    log_returns = np.log(close / np.roll(close, 1))
    log_returns[0] = 0.0

    # -----------------------------
    # OPTIMISATION (NUMPY ONLY)
    # -----------------------------
    best_return = -np.inf
    best_params = None

    for ma1 in range(1, 20):
        for ma2 in range(20, 40):
            for ma3 in range(40, 81):

                max_ma = np.maximum.reduce(
                    [ma_matrix[:, ma1], ma_matrix[:, ma2], ma_matrix[:, ma3]]
                )
                max_ma_prev = np.maximum.reduce(
                    [ma_prev_matrix[:, ma1], ma_prev_matrix[:, ma2], ma_prev_matrix[:, ma3]]
                )

                min_ma = np.minimum.reduce(
                    [ma_matrix[:, ma1], ma_matrix[:, ma2], ma_matrix[:, ma3]]
                )
                min_ma_prev = np.minimum.reduce(
                    [ma_prev_matrix[:, ma1], ma_prev_matrix[:, ma2], ma_prev_matrix[:, ma3]]
                )

                long_entry = (close > max_ma) & (close_prev <= max_ma_prev)
                short_entry = (close < min_ma) & (close_prev >= min_ma_prev)

                signal = np.zeros(len(close))
                signal[long_entry] = 1
                signal[short_entry] = -1

                # Position (manual forward fill)
                position = np.zeros(len(signal))
                for i in range(1, len(signal)):
                    position[i] = signal[i] if signal[i] != 0 else position[i - 1]

                strategy_returns = log_returns * np.roll(position, 1)
                total_return = np.nansum(strategy_returns)

                if total_return > best_return:
                    best_return = total_return
                    best_params = (ma1, ma2, ma3)

    final_results.append({
        "Ticker": ticker,
        "MA_1": best_params[0],
        "MA_2": best_params[1],
        "MA_3": best_params[2],
        "Total_Log_Return_5Y": best_return
    })

# ==========================================================
# RESULTS
# ==========================================================

results_df = pd.DataFrame(final_results)
results_df = results_df.sort_values(
    by="Total_Log_Return_5Y", ascending=False
).reset_index(drop=True)

print("\nBEST THREE-MA STRATEGY (NUMPY-ONLY, 5 YEARS):\n")
print(results_df}
