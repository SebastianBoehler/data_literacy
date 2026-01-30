#!/usr/bin/env python3
"""
Quantile Analysis of Delay Distribution

Shows delay percentiles (50th, 75th, 90th, 95th, 99th) with bootstrap CIs.
More robust than means for skewed data. Compares before/after schedule change.

Key insight: "90% of buses arrive within X minutes of schedule"

Output: outputs/quantile_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Increase font sizes for readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "quantile_analysis.png"

# Constants
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")
QUANTILES = [50, 75, 90, 95, 99]
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def bootstrap_quantile_ci(data, q, n_boot=N_BOOTSTRAP):
    """Calculate bootstrap CI for a quantile."""
    boot_quantiles = [np.percentile(np.random.choice(data, len(data), replace=True), q) 
                      for _ in range(n_boot)]
    return np.percentile(boot_quantiles, 2.5), np.percentile(boot_quantiles, 97.5)


def main():
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    delays = df['delay_minutes'].dropna().values

    pre = df[df['timestamp'] < SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna().values
    post = df[df['timestamp'] >= SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna().values

    print(f"Total: {len(delays):,}, Pre: {len(pre):,}, Post: {len(post):,}")

    # Calculate quantiles with CIs
    all_stats = []
    for q in QUANTILES:
        val = np.percentile(delays, q)
        ci_low, ci_high = bootstrap_quantile_ci(delays, q)
        all_stats.append({'quantile': q, 'value': val, 'ci_low': ci_low, 'ci_high': ci_high, 'period': 'All'})

    for q in QUANTILES:
        val_pre = np.percentile(pre, q)
        ci_low_pre, ci_high_pre = bootstrap_quantile_ci(pre, q)
        all_stats.append({'quantile': q, 'value': val_pre, 'ci_low': ci_low_pre, 'ci_high': ci_high_pre, 'period': 'Before'})

        val_post = np.percentile(post, q)
        ci_low_post, ci_high_post = bootstrap_quantile_ci(post, q)
        all_stats.append({'quantile': q, 'value': val_post, 'ci_low': ci_low_post, 'ci_high': ci_high_post, 'period': 'After'})

    stats_df = pd.DataFrame(all_stats)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Quantile values with CI (all data)
    ax1 = axes[0]
    all_data = stats_df[stats_df['period'] == 'All']
    x = range(len(QUANTILES))
    ax1.bar(x, all_data['value'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.errorbar(x, all_data['value'],
                 yerr=[all_data['value'] - all_data['ci_low'], all_data['ci_high'] - all_data['value']],
                 fmt='none', color='black', capsize=5, capthick=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{q}th' for q in QUANTILES])
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Delay (minutes)')
    ax1.set_title('Delay Quantiles with 95% Bootstrap CI', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(all_data.iterrows()):
        ax1.text(i, row['value'] + 0.5, f'{row["value"]:.1f}', ha='center', fontsize=9)

    ax1.text(0.02, 0.98, f'n = {len(delays):,}', transform=ax1.transAxes, fontsize=9, va='top', color='gray')
    ax1.text(0.98, 0.98, f'90% of buses arrive\nwithin {np.percentile(delays, 90):.1f} min of schedule',
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: Pre vs Post comparison
    ax2 = axes[1]
    pre_data = stats_df[stats_df['period'] == 'Before']
    post_data = stats_df[stats_df['period'] == 'After']

    width = 0.35
    x = np.arange(len(QUANTILES))

    ax2.bar(x - width / 2, pre_data['value'].values, width, label='Before', color='steelblue', alpha=0.7)
    ax2.bar(x + width / 2, post_data['value'].values, width, label='After', color='darkorange', alpha=0.7)

    ax2.errorbar(x - width / 2, pre_data['value'].values,
                 yerr=[pre_data['value'].values - pre_data['ci_low'].values,
                       pre_data['ci_high'].values - pre_data['value'].values],
                 fmt='none', color='black', capsize=3)
    ax2.errorbar(x + width / 2, post_data['value'].values,
                 yerr=[post_data['value'].values - post_data['ci_low'].values,
                       post_data['ci_high'].values - post_data['value'].values],
                 fmt='none', color='black', capsize=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{q}th' for q in QUANTILES])
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Delay (minutes)')
    ax2.set_title('Delay Quantiles: Before vs After Schedule Change', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Print findings
    print(f'\nKey findings:')
    print(f'  Median (50th): {np.percentile(delays, 50):.1f} min')
    print(f'  90th percentile: {np.percentile(delays, 90):.1f} min')
    print(f'\nSchedule change effect on 90th percentile:')
    print(f'  Before: {np.percentile(pre, 90):.1f} min → After: {np.percentile(post, 90):.1f} min')
    print(f'  Improvement: {np.percentile(pre, 90) - np.percentile(post, 90):.1f} min')


if __name__ == "__main__":
    main()
