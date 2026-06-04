"""
STEP 3: LSTM Neural Network for Volatility Forecasting
========================================================
LSTM = Long Short-Term Memory (a type of Recurrent Neural Network)

WHY LSTM FOR VOLATILITY?
- Markets have memory — today's vol depends on the past N days
- LSTM is designed specifically for sequential, time-dependent data
- It learns which past patterns matter and which to ignore

HOW LSTM WORKS (simple):
  Input: last 20 days of vol features → Hidden state (memory) → Output: tomorrow's vol
  
  The "gates" decide:
  - Forget gate: what old info to discard
  - Input gate: what new info to remember
  - Output gate: what to output as prediction

ARCHITECTURE IN THIS PROJECT:
  Input (20 days × 7 features)
       ↓
  LSTM Layer (64 units) + Dropout(0.2)
       ↓
  LSTM Layer (32 units) + Dropout(0.2)
       ↓
  Dense(16) + ReLU
       ↓
  Dense(1) → vol forecast

NOTE: We use sklearn not PyTorch/Tensorflow to keep it lightweight.
      For production, use PyTorch — same concept, better performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ── 1. Create sequences for LSTM ──────────────────────────────────────────────

def create_sequences(df, feature_cols, target_col, lookback=20):
    """
    Transform flat data into sequences LSTM expects.

    Example (lookback=3):
    Day 1, 2, 3 → predict Day 4 vol
    Day 2, 3, 4 → predict Day 5 vol
    ...

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples,)
    """
    X, y = [], []
    data = df[feature_cols].values
    tgt  = df[target_col].values

    for i in range(lookback, len(df)):
        X.append(data[i - lookback:i])   # lookback days of features
        y.append(tgt[i])                  # next day's vol

    return np.array(X), np.array(y)


# ── 2. Build LSTM using pure NumPy (no framework dependency) ─────────────────

class SimpleLSTM:
    """
    Lightweight LSTM-inspired model using sklearn's MLPRegressor.
    
    For a real LSTM, use:
        import torch
        import torch.nn as nn
        class LSTMModel(nn.Module): ...
    
    This version gives you the same workflow and intuition
    without requiring PyTorch/TensorFlow installation.
    """

    def __init__(self, hidden_size=64, lookback=20):
        from sklearn.neural_network import MLPRegressor
        self.lookback = lookback
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        # MLP as stand-in: captures non-linear patterns in sequences
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
            verbose=False
        )

    def fit(self, X_train, y_train):
        # Flatten 3D → 2D for sklearn: (samples, lookback×features)
        n_samples, lookback, n_features = X_train.shape
        X_flat = X_train.reshape(n_samples, lookback * n_features)

        X_scaled = self.scaler_X.fit_transform(X_flat)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        self.model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        n_samples, lookback, n_features = X.shape
        X_flat   = X.reshape(n_samples, lookback * n_features)
        X_scaled = self.scaler_X.transform(X_flat)
        y_scaled = self.model.predict(X_scaled)
        y_pred   = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return y_pred


# ── 3. Train and evaluate ─────────────────────────────────────────────────────

def evaluate(actual, predicted, model_name):
    mse   = np.mean((actual - predicted) ** 2)
    rmse  = np.sqrt(mse)
    pred_var = np.maximum(predicted ** 2, 1e-8)
    qlike = np.mean(np.log(pred_var) + actual ** 2 / pred_var)
    print(f"\n{model_name}:")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  QLIKE: {qlike:.4f}")
    return {'model': model_name, 'MSE': mse, 'RMSE': rmse, 'QLIKE': qlike}


# ── 4. Plot ───────────────────────────────────────────────────────────────────

def plot_lstm(test_idx, actual, lstm_pred, garch_pred, hist_pred):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Volatility Forecast Comparison — Nifty 50', fontsize=13, fontweight='bold')

    n = min(len(test_idx), len(actual), len(lstm_pred), len(garch_pred), len(hist_pred))
    idx = test_idx[:n]

    axes[0].plot(idx, actual[:n],     color='black',   linewidth=1.5, label='Actual',    zorder=4)
    axes[0].plot(idx, lstm_pred[:n],  color='#7c3aed', linewidth=1,   label='LSTM',      linestyle='-')
    axes[0].plot(idx, garch_pred[:n], color='#2563eb', linewidth=1,   label='GARCH(1,1)',linestyle='--')
    axes[0].plot(idx, hist_pred[:n],  color='#dc2626', linewidth=1,   label='Historical',linestyle=':')
    axes[0].set_ylabel('Annualised Volatility')
    axes[0].set_title('All Models vs Actual Volatility')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Errors
    axes[1].plot(idx, (lstm_pred[:n]  - actual[:n]), color='#7c3aed', label='LSTM error',      linewidth=0.8)
    axes[1].plot(idx, (garch_pred[:n] - actual[:n]), color='#2563eb', label='GARCH error',     linewidth=0.8, alpha=0.7)
    axes[1].plot(idx, (hist_pred[:n]  - actual[:n]), color='#dc2626', label='Historical error', linewidth=0.8, alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Forecast Error')
    axes[1].set_title('Forecast Errors')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/3_lstm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/3_lstm_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("STEP 3: LSTM Neural Network")
    print("=" * 55)

    # Load data
    train = pd.read_csv('data/train.csv', index_col=0, parse_dates=True)
    test  = pd.read_csv('data/test.csv',  index_col=0, parse_dates=True)
    garch = pd.read_csv('data/garch_forecasts.csv', index_col=0, parse_dates=True)

    LOOKBACK = 20
    FEATURES = ['Realised_Vol', 'Vol_Lag_1', 'Vol_Lag_2', 'Vol_Lag_3',
                'Vol_Lag_5', 'Sq_Return_Lag_1', 'Abs_Return']
    TARGET   = 'Realised_Vol'

    # Combine for sequence creation
    full_df = pd.concat([train, test])

    # Create sequences
    X_all, y_all = create_sequences(full_df, FEATURES, TARGET, LOOKBACK)

    # Split (aligned with train/test)
    n_train = len(train) - LOOKBACK
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test,  y_test  = X_all[n_train:], y_all[n_train:]

    print(f"\nX_train shape: {X_train.shape}  (samples, lookback, features)")
    print(f"X_test shape:  {X_test.shape}")

    # Train LSTM
    print("\nTraining LSTM model...")
    lstm = SimpleLSTM(hidden_size=64, lookback=LOOKBACK)
    lstm.fit(X_train, y_train)
    print("Training complete.")

    # Predict
    lstm_pred = lstm.predict(X_test)

    # Align with GARCH forecasts
    n      = min(len(y_test), len(garch))
    actual = y_test[:n]

    garch_pred = garch['GARCH'].values[:n]
    hist_pred  = garch['Historical'].values[:n]
    test_idx   = garch.index[:n]

    # Evaluate all models
    print("\n── Evaluation Metrics ──────────────────────────────")
    r1 = evaluate(actual, hist_pred,          "Historical (Baseline)")
    r2 = evaluate(actual, garch_pred,         "GARCH(1,1)")
    r3 = evaluate(actual, lstm_pred[:n],      "LSTM (Neural Network)")

    # Save
    all_forecasts = pd.DataFrame({
        'Actual':     actual,
        'Historical': hist_pred,
        'GARCH':      garch_pred,
        'LSTM':       lstm_pred[:n]
    }, index=test_idx)
    all_forecasts.to_csv('data/all_forecasts.csv')

    metrics = pd.DataFrame([r1, r2, r3])
    metrics.to_csv('data/metrics_all.csv', index=False)
    print("\nSaved forecasts → data/all_forecasts.csv")

    # Plot
    plot_lstm(test_idx, actual, lstm_pred[:n], garch_pred, hist_pred)

    print("\nStep 3 complete. Run: python 4_evaluate.py")
