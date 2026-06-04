"""
STEP 4: Final Evaluation & Summary Dashboard
=============================================
This is the output you show in interviews, on GitHub, and to RIA clients.

WHAT YOU'LL GET:
- Model comparison table (RMSE, QLIKE, % improvement over baseline)
- 4-panel summary chart
- VaR backtest: does better vol forecast → better risk management?
- Saved PNG suitable for GitHub README
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# ── 1. VaR Backtest ───────────────────────────────────────────────────────────

def compute_var(vol_forecast, confidence=0.99, holding_period=1):
    """
    Value at Risk using parametric method:
    VaR = z × σ × √holding_period
    
    z = 2.326 for 99% confidence
    σ = daily volatility (annualised vol / sqrt(252))
    
    This is exactly how banks compute VaR for their trading books.
    Better vol forecast → more accurate VaR → less capital wasted.
    """
    from scipy.stats import norm
    z         = norm.ppf(confidence)
    daily_vol = vol_forecast / np.sqrt(252)
    var       = z * daily_vol * np.sqrt(holding_period)
    return var


def var_backtest(actual_returns, var_forecast, confidence=0.99):
    """
    Count how many times actual loss exceeded VaR forecast (violations).
    
    Basel III rule: at 99% confidence, violations should be ~1% of days.
    Too many violations = model under-estimates risk (dangerous!)
    Too few violations = model over-estimates risk (wastes capital)
    
    This is called the Kupiec Test in real risk management.
    """
    violations     = actual_returns < -var_forecast
    n_violations   = violations.sum()
    violation_rate = n_violations / len(actual_returns)
    expected_rate  = 1 - confidence

    print(f"  Violations:    {n_violations} / {len(actual_returns)} days")
    print(f"  Violation rate: {violation_rate:.2%}  (expected: {expected_rate:.2%})")

    if violation_rate < expected_rate * 0.5:
        print("  ⚠ Model OVER-estimates risk (too conservative — wastes capital)")
    elif violation_rate > expected_rate * 1.5:
        print("  ⚠ Model UNDER-estimates risk (too optimistic — dangerous!)")
    else:
        print("  ✓ VaR model is well-calibrated")

    return violation_rate


# ── 2. Summary metrics table ──────────────────────────────────────────────────

def print_summary_table(metrics_df, baseline_rmse):
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'RMSE':>10} {'QLIKE':>10} {'Improvement':>12}")
    print("-" * 60)
    for _, row in metrics_df.iterrows():
        improvement = (baseline_rmse - row['RMSE']) / baseline_rmse * 100
        marker = " ← best" if row['RMSE'] == metrics_df['RMSE'].min() else ""
        print(f"{row['model']:<22} {row['RMSE']:>10.6f} {row['QLIKE']:>10.4f} {improvement:>11.1f}%{marker}")
    print("=" * 60)


# ── 3. Final 4-panel chart ────────────────────────────────────────────────────

def plot_final_dashboard(forecasts_df, metrics_df, returns_series):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    actual     = forecasts_df['Actual'].values
    lstm_pred  = forecasts_df['LSTM'].values
    garch_pred = forecasts_df['GARCH'].values
    hist_pred  = forecasts_df['Historical'].values
    idx        = forecasts_df.index

    # ── Panel 1: All forecasts ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx, actual,     color='black',   lw=1.5, label='Realised Vol',  zorder=4)
    ax1.plot(idx, lstm_pred,  color='#7c3aed', lw=1.0, label='LSTM',          ls='-')
    ax1.plot(idx, garch_pred, color='#2563eb', lw=1.0, label='GARCH(1,1)',    ls='--')
    ax1.plot(idx, hist_pred,  color='#dc2626', lw=1.0, label='Historical',    ls=':')
    ax1.fill_between(idx, actual, alpha=0.08, color='black')
    ax1.set_title('Nifty 50 Volatility Forecasting — All Models', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Annualised Volatility')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # ── Panel 2: RMSE bar chart ──
    ax2 = fig.add_subplot(gs[1, 0])
    models = metrics_df['model'].tolist()
    rmses  = metrics_df['RMSE'].tolist()
    colors = ['#dc2626', '#2563eb', '#7c3aed']
    bars   = ax2.bar(models, rmses, color=colors, width=0.5, edgecolor='white')
    ax2.set_title('RMSE by Model\n(lower is better)', fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)

    # ── Panel 3: VaR comparison ──
    ax3 = fig.add_subplot(gs[1, 1])
    test_len    = len(forecasts_df)
    ret_aligned = returns_series.iloc[-test_len:].values if len(returns_series) >= test_len else returns_series.values[-test_len:]
    n           = min(len(ret_aligned), len(lstm_pred), len(garch_pred))

    lstm_var  = compute_var(lstm_pred[:n])
    garch_var = compute_var(garch_pred[:n])
    rets      = ret_aligned[:n]

    ax3.plot(idx[:n], -lstm_var,          color='#7c3aed', lw=1, label='LSTM 99% VaR',   ls='-')
    ax3.plot(idx[:n], -garch_var,         color='#2563eb', lw=1, label='GARCH 99% VaR',  ls='--')
    ax3.scatter(idx[:n][rets < -lstm_var], rets[rets < -lstm_var],
                color='#7c3aed', s=15, zorder=5, label='LSTM violation', alpha=0.7)
    ax3.scatter(idx[:n][rets < -garch_var], rets[rets < -garch_var],
                color='#2563eb', s=15, marker='x', zorder=5, label='GARCH violation', alpha=0.7)
    ax3.bar(idx[:n], rets, color='#94a3b8', alpha=0.4, label='Daily return')
    ax3.set_title('VaR Backtest (99% confidence)\nDots = VaR violations', fontweight='bold')
    ax3.set_ylabel('Daily Return / VaR')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.suptitle('Volatility Forecasting Project — Nifty 50\nQuantitative Finance + ML',
                 fontsize=12, fontweight='bold', y=1.01)

    plt.savefig('outputs/4_final_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/4_final_dashboard.png  ← use this in your GitHub README")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: Final Evaluation & Dashboard")
    print("=" * 60)

    # Load results
    forecasts = pd.read_csv('data/all_forecasts.csv',  index_col=0, parse_dates=True)
    metrics   = pd.read_csv('data/metrics_all.csv')
    full_df   = pd.read_csv('data/nifty_features.csv', index_col=0, parse_dates=True)

    # Summary table
    baseline_rmse = metrics.loc[metrics['model'] == 'Historical (Baseline)', 'RMSE'].values[0]
    print_summary_table(metrics, baseline_rmse)

    # VaR backtest
    print("\n── VaR Backtest (99% confidence) ─────────────────────")
    n     = len(forecasts)
    rets  = full_df['Log_Return'].iloc[-n:]

    print("\nLSTM VaR:")
    lstm_viol  = var_backtest(rets.values, compute_var(forecasts['LSTM'].values))
    print("\nGARCH VaR:")
    garch_viol = var_backtest(rets.values, compute_var(forecasts['GARCH'].values))

    # Final dashboard
    plot_final_dashboard(forecasts, metrics, full_df['Log_Return'])

    # Key takeaways
    best    = metrics.loc[metrics['RMSE'].idxmin(), 'model']
    improve = (baseline_rmse - metrics['RMSE'].min()) / baseline_rmse * 100

    print("\n── Key Takeaways ──────────────────────────────────────")
    print(f"  Best model:    {best}")
    print(f"  Improvement:   {improve:.1f}% over historical baseline")
    print(f"\n  What this means for your RIA business:")
    print(f"  → Use GARCH/LSTM vol forecast as input to stress tests")
    print(f"  → Tell RIA clients: 'Current Nifty vol regime = X%'")
    print(f"  → Trigger rebalance alert when vol crosses threshold")

