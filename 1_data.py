"""
STEP 1: Data Preparation
========================
Goal: Load Nifty 50 data, compute log returns and realised volatility.

What is realised volatility?
- We can't directly observe volatility — we estimate it from past returns
- Rolling 21-day std of returns × sqrt(252) = annualised realised vol
- This becomes our TARGET variable that GARCH and LSTM will try to predict
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Generate realistic Nifty 50 data ───────────────────────────────────────
# NOTE: In production, replace this with:
#   import yfinance as yf
#   df = yf.download("^NSEI", start="2018-01-01", end="2024-12-31")
#   df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
# ─────────────────────────────────────────────────────────────────────────────

def generate_nifty_data(n_days=1500, seed=42):
    """
    Simulate Nifty 50 using GBM + GARCH volatility clustering.
    Realistic params: 12% annual return, ~18% annual vol, base ~10000.
    """
    np.random.seed(seed)

    dt      = 1 / 252
    mu      = 0.12       # annual drift
    omega   = 0.000002   # GARCH constant
    alpha   = 0.09       # ARCH term (shock impact)
    beta    = 0.90       # GARCH term (vol persistence)
    S0      = 10000      # starting price

    returns = np.zeros(n_days)
    vol     = np.zeros(n_days)
    vol[0]  = 0.18 * np.sqrt(dt)

    for t in range(1, n_days):
        z          = np.random.normal()
        returns[t] = mu * dt + vol[t-1] * z
        vol[t]     = np.sqrt(omega + alpha * returns[t-1]**2 + beta * vol[t-1]**2)

    prices = S0 * np.exp(np.cumsum(returns))
    dates  = pd.bdate_range(start='2018-01-01', periods=n_days)

    return pd.DataFrame({'Close': prices, 'Log_Return': returns}, index=dates)


# ── 2. Compute features ───────────────────────────────────────────────────────

def add_features(df, window=21):
    """
    Add volatility and lagged features.

    Key concepts:
    - Log return: ln(P_t / P_{t-1}) — better than % return for modelling
    - Realised vol: rolling std of returns × sqrt(252) — annualised
    - window=21: ~1 trading month, standard in risk management
    """
    df = df.copy()

    # Squared returns — proxy for variance (used in GARCH)
    df['Squared_Return'] = df['Log_Return'] ** 2

    # Realised volatility (our TARGET to predict)
    df['Realised_Vol'] = (
        df['Log_Return']
        .rolling(window)
        .std()
        * np.sqrt(252)
    )

    # Lagged realised vol (features for ML model)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Vol_Lag_{lag}'] = df['Realised_Vol'].shift(lag)

    # Lagged squared returns (features)
    for lag in [1, 2, 3]:
        df[f'Sq_Return_Lag_{lag}'] = df['Squared_Return'].shift(lag)

    # Absolute return (another vol proxy)
    df['Abs_Return'] = df['Log_Return'].abs()

    # Drop NaN rows from rolling window + lags
    df = df.dropna()

    return df


# ── 3. Train / test split ─────────────────────────────────────────────────────

def split_data(df, test_ratio=0.2):
    """
    Time series split — NEVER shuffle financial data.
    Train on past, test on future. Always.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train     = df.iloc[:split_idx]
    test      = df.iloc[split_idx:]
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} days)")
    print(f"Test:  {test.index[0].date()}  → {test.index[-1].date()}   ({len(test)} days)")
    return train, test


# ── 4. Plot & save ────────────────────────────────────────────────────────────

def plot_data(df, train, test):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Nifty 50 — Data Overview', fontsize=14, fontweight='bold')

    # Price
    axes[0].plot(df.index, df['Close'], color='#2563eb', linewidth=1)
    axes[0].set_title('Price')
    axes[0].set_ylabel('Index Level')
    axes[0].grid(alpha=0.3)

    # Log returns
    axes[1].plot(df.index, df['Log_Return'], color='#7c3aed', linewidth=0.6, alpha=0.8)
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_title('Daily Log Returns')
    axes[1].set_ylabel('Return')
    axes[1].grid(alpha=0.3)

    # Realised volatility with train/test split
    axes[2].plot(train.index, train['Realised_Vol'], color='#059669', linewidth=1,
                 label='Train')
    axes[2].plot(test.index,  test['Realised_Vol'],  color='#dc2626', linewidth=1,
                 label='Test')
    axes[2].axvline(test.index[0], color='black', linewidth=1, linestyle='--',
                    label='Train/Test split')
    axes[2].set_title('21-Day Realised Volatility (Annualised) — TARGET variable')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/1_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/1_data_overview.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("STEP 1: Data Preparation")
    print("=" * 55)

    # Generate data
    raw_df = generate_nifty_data(n_days=1500)
    print(f"\nRaw data: {len(raw_df)} trading days")

    # Add features
    df = add_features(raw_df, window=21)
    print(f"After feature engineering: {len(df)} rows, {len(df.columns)} columns")

    # Summary stats
    print(f"\nAnnualised vol range:  {df['Realised_Vol'].min():.2%} — {df['Realised_Vol'].max():.2%}")
    print(f"Mean realised vol:     {df['Realised_Vol'].mean():.2%}")

    # Split
    print("\nTrain / Test Split:")
    train, test = split_data(df)

    # Save
    df.to_csv('data/nifty_features.csv')
    train.to_csv('data/train.csv')
    test.to_csv('data/test.csv')
    print("\nSaved data to data/")

    # Plot
    plot_data(df, train, test)

    print("\nStep 1 complete. Run: python 2_garch.py")
