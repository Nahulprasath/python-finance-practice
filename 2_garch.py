"""
STEP 2: GARCH(1,1) Volatility Model
=====================================
GARCH = Generalised Autoregressive Conditional Heteroskedasticity
(Don't worry about the name — focus on what it does)

WHAT IT DOES:
- Models the fact that volatility clusters — calm periods follow calm,
  turbulent periods follow turbulent (you can see this in any market)

GARCH(1,1) FORMULA:
  σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
  
  where:
  σ²_t        = today's variance (what we're predicting)
  ω           = long-run average variance (constant)
  α × ε²_{t-1} = impact of yesterday's shock (ARCH term)
  β × σ²_{t-1} = persistence of yesterday's variance (GARCH term)
  
  α + β < 1   = stationary (vol eventually reverts to mean)
  α + β → 1   = vol is very persistent (common in real markets)

WHY EVERY RISK DESK USES IT:
- VaR calculations require a vol forecast
- Options desks use it for implied vs realised vol comparison
- Stress testing: "what if vol spikes to X?" — GARCH quantifies this

WHAT YOU'LL PRODUCE:
- Fitted GARCH model on train data
- 1-day-ahead vol forecasts on test data
- Evaluation metrics: MSE, RMSE, QLIKE (industry standard)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model


# ── 1. Fit GARCH(1,1) ─────────────────────────────────────────────────────────

def fit_garch(train_returns):
    """
    Fit GARCH(1,1) with Normal distribution.

    Note: returns are scaled ×100 for numerical stability
    (arch library convention — doesn't affect results)
    """
    returns_scaled = train_returns * 100

    model = arch_model(
        returns_scaled,
        vol='Garch',    # GARCH variance model
        p=1,            # 1 lag for squared returns (ARCH term)
        q=1,            # 1 lag for variance (GARCH term)
        dist='Normal',  # innovation distribution
        mean='Constant' # constant mean (μ)
    )

    result = model.fit(disp='off', show_warning=False)
    return model, result


def print_garch_params(result):
    """Explain what each parameter means."""
    params = result.params
    print("\nGARCH(1,1) Parameters:")
    print(f"  μ (mean return):    {params['mu']:.4f}")
    print(f"  ω (base variance):  {params['omega']:.6f}")
    print(f"  α (shock impact):   {params['alpha[1]']:.4f}  ← how much yesterday's shock matters")
    print(f"  β (vol persistence):{params['beta[1]']:.4f}  ← how long shocks last")
    print(f"  α + β:              {params['alpha[1]'] + params['beta[1]']:.4f}  ← < 1 means mean-reverting")
    print(f"\n  Log-likelihood: {result.loglikelihood:.2f}")
    print(f"  AIC:            {result.aic:.2f}")


# ── 2. Rolling forecast on test data ─────────────────────────────────────────

def rolling_garch_forecast(train_returns, test_returns):
    """
    Walk-forward (rolling) forecast:
    - At each test day t, refit GARCH on all data up to t-1
    - Forecast vol for day t
    - Move forward one day

    This is how it works in REAL production at banks —
    you never train on future data.
    """
    print("\nRunning rolling GARCH forecasts (this takes ~30 seconds)...")

    all_returns  = pd.concat([train_returns, test_returns])
    forecasts    = []
    n_train      = len(train_returns)

    # Use expanding window (retrain every 5 days for speed)
    step = 5
    prev_forecast = None

    for i in range(len(test_returns)):
        idx = n_train + i

        # Retrain every `step` days
        if i % step == 0:
            window_returns = all_returns.iloc[:idx] * 100
            try:
                m = arch_model(window_returns, vol='Garch', p=1, q=1,
                               dist='Normal', mean='Constant')
                r = m.fit(disp='off', show_warning=False, options={'maxiter': 200})
                forecast     = r.forecast(horizon=1, reindex=False)
                vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
                prev_forecast = vol_forecast * np.sqrt(252)  # annualise
            except:
                prev_forecast = prev_forecast or 0.18

        forecasts.append(prev_forecast)

        if (i + 1) % 50 == 0:
            print(f"  Forecasted {i+1}/{len(test_returns)} days")

    return np.array(forecasts)


# ── 3. Baseline: rolling historical vol ──────────────────────────────────────

def historical_vol_baseline(returns, window=21):
    """
    Simplest benchmark: just use the last 21 days of realised vol
    as tomorrow's vol forecast. Surprisingly hard to beat!
    """
    return returns.rolling(window).std() * np.sqrt(252)


# ── 4. Evaluate forecasts ─────────────────────────────────────────────────────

def evaluate(actual, predicted, model_name):
    """
    Three metrics used in industry:
    - MSE:   Mean Squared Error (penalises large errors)
    - RMSE:  Root MSE (same units as vol, easier to interpret)
    - QLIKE: Quasi-likelihood loss — standard in vol forecasting research
             QLIKE = log(σ²) + realised_var/σ²
             Better than MSE for fat-tailed distributions
    """
    mse   = np.mean((actual - predicted) ** 2)
    rmse  = np.sqrt(mse)

    # QLIKE (Patton 2011) — use variance, not vol
    actual_var    = actual ** 2
    predicted_var = predicted ** 2
    predicted_var = np.maximum(predicted_var, 1e-8)  # avoid log(0)
    qlike = np.mean(np.log(predicted_var) + actual_var / predicted_var)

    print(f"\n{model_name}:")
    print(f"  RMSE:  {rmse:.6f}  (lower is better)")
    print(f"  QLIKE: {qlike:.4f}  (lower is better)")
    return {'model': model_name, 'MSE': mse, 'RMSE': rmse, 'QLIKE': qlike}


# ── 5. Plot ───────────────────────────────────────────────────────────────────

def plot_garch(test_df, garch_forecast, hist_forecast, split_date):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('GARCH(1,1) vs Historical Vol — Nifty 50', fontsize=13, fontweight='bold')

    actual = test_df['Realised_Vol'].values

    # Top: forecast comparison
    axes[0].plot(test_df.index, actual,         color='black',   linewidth=1.2,
                 label='Realised Vol (actual)',  zorder=3)
    axes[0].plot(test_df.index, garch_forecast, color='#2563eb', linewidth=1,
                 label='GARCH(1,1) forecast',   linestyle='--')
    axes[0].plot(test_df.index, hist_forecast,  color='#dc2626', linewidth=1,
                 label='Historical (21d) baseline', linestyle=':')
    axes[0].set_ylabel('Annualised Volatility')
    axes[0].set_title('Volatility Forecasts vs Actual')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Bottom: forecast error
    garch_error = garch_forecast - actual
    hist_error  = hist_forecast  - actual
    axes[1].plot(test_df.index, garch_error, color='#2563eb', linewidth=0.8,
                 label='GARCH error', alpha=0.8)
    axes[1].plot(test_df.index, hist_error,  color='#dc2626', linewidth=0.8,
                 label='Historical error', alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Forecast Error')
    axes[1].set_title('Forecast Errors (Predicted − Actual)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/2_garch_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/2_garch_forecast.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("STEP 2: GARCH(1,1) Model")
    print("=" * 55)

    # Load data
    train = pd.read_csv('data/train.csv', index_col=0, parse_dates=True)
    test  = pd.read_csv('data/test.csv',  index_col=0, parse_dates=True)

    train_ret = train['Log_Return']
    test_ret  = test['Log_Return']
    actual    = test['Realised_Vol'].values

    # Fit and print GARCH on train
    print("\nFitting GARCH(1,1) on training data...")
    model, result = fit_garch(train_ret)
    print_garch_params(result)

    # Rolling forecasts on test
    garch_forecast = rolling_garch_forecast(train_ret, test_ret)

    # Baseline
    all_returns = pd.concat([train_ret, test_ret])
    hist_vol    = historical_vol_baseline(all_returns)
    hist_forecast = hist_vol.loc[test.index].values

    # Align lengths (handle any NaN at start)
    min_len        = min(len(actual), len(garch_forecast), len(hist_forecast))
    actual         = actual[-min_len:]
    garch_forecast = garch_forecast[-min_len:]
    hist_forecast  = hist_forecast[-min_len:]
    test_aligned   = test.iloc[-min_len:]

    # Evaluate
    print("\n── Evaluation Metrics ──────────────────────────────")
    r1 = evaluate(actual, hist_forecast,  "Historical (Baseline)")
    r2 = evaluate(actual, garch_forecast, "GARCH(1,1)")

    # Save forecasts for comparison in step 4
    results_df = pd.DataFrame({
        'Actual':      actual,
        'Historical':  hist_forecast,
        'GARCH':       garch_forecast
    }, index=test_aligned.index)
    results_df.to_csv('data/garch_forecasts.csv')

    # Save metrics
    metrics = pd.DataFrame([r1, r2])
    metrics.to_csv('data/metrics_garch.csv', index=False)

    # Plot
    plot_garch(test_aligned, garch_forecast, hist_forecast, test.index[0])

    print("\nStep 2 complete. Run: python 3_lstm.py")
