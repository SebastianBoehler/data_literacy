#!/usr/bin/env python3
"""
Day of Week Effect on Late Rate

Compares late rates across days of the week and weekday vs weekend.
Shows effect size with bootstrap confidence intervals.

Output: outputs/day_of_week_effect.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.plot_config import apply_style, STYLE

apply_style()

SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "day_of_week_effect.png"

LATE_THRESHOLD = 2
N_BOOTSTRAP = 500
RANDOM_SEED = 42
DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def bootstrap_ci(data, func, n_boot=N_BOOTSTRAP):
    stats = [func(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def main():
    np.random.seed(RANDOM_SEED)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6])

    # By day of week
    day_stats = []
    for dow in range(7):
        day_data = df[df['dayofweek'] == dow]['delay_minutes'].dropna().values
        if len(day_data) > 100:
            late_rate = (day_data > LATE_THRESHOLD).mean()
            ci_low, ci_high = bootstrap_ci(day_data, lambda x: (x > LATE_THRESHOLD).mean())
            day_stats.append({
                'day': DAY_NAMES[dow],
                'dow': dow,
                'n': len(day_data),
                'late_rate': late_rate,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'is_weekend': dow >= 5
            })

    day_df = pd.DataFrame(day_stats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: By day of week
    ax1 = axes[0]
    colors = ['steelblue' if not w else 'darkorange' for w in day_df['is_weekend']]
    ax1.bar(day_df['day'], day_df['late_rate'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(range(len(day_df)), day_df['late_rate'] * 100,
                 yerr=[(day_df['late_rate'] - day_df['ci_low']) * 100,
                       (day_df['ci_high'] - day_df['late_rate']) * 100],
                 fmt='none', color='black', capsize=4)

    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Late Rate (%)')
    ax1.set_title('Late Rate by Day of Week (blue=weekday, orange=weekend)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, row in day_df.iterrows():
        ax1.text(i, row['late_rate'] * 100 + 1, f'n={row["n"]:,}', ha='center', fontsize=8, color='gray')

    # Right: Weekday vs Weekend
    ax2 = axes[1]
    weekday_data = df[~df['is_weekend']]['delay_minutes'].dropna().values
    weekend_data = df[df['is_weekend']]['delay_minutes'].dropna().values

    wd_rate = (weekday_data > LATE_THRESHOLD).mean()
    we_rate = (weekend_data > LATE_THRESHOLD).mean()
    wd_ci = bootstrap_ci(weekday_data, lambda x: (x > LATE_THRESHOLD).mean())
    we_ci = bootstrap_ci(weekend_data, lambda x: (x > LATE_THRESHOLD).mean())

    x = [0, 1]
    rates = [wd_rate * 100, we_rate * 100]
    ci_low = [wd_ci[0] * 100, we_ci[0] * 100]
    ci_high = [wd_ci[1] * 100, we_ci[1] * 100]

    ax2.bar(x, rates, color=['steelblue', 'darkorange'], alpha=0.7, edgecolor='black')
    ax2.errorbar(x, rates, yerr=[[r - l for r, l in zip(rates, ci_low)],
                                  [h - r for r, h in zip(rates, ci_high)]],
                 fmt='none', color='black', capsize=5, capthick=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Weekday', 'Weekend'])
    ax2.set_ylabel('Late Rate (%)')
    ax2.set_title('Weekday vs Weekend Late Rate with 95% CI', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    ax2.text(0, wd_rate * 100 + 1.5, f'n={len(weekday_data):,}', ha='center', fontsize=9, color='gray')
    ax2.text(1, we_rate * 100 + 1.5, f'n={len(weekend_data):,}', ha='center', fontsize=9, color='gray')

    diff = wd_rate - we_rate
    ax2.text(0.5, max(rates) * 0.5, f'Δ = {diff * 100:.1f} pp', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    print(f"\nWeekday vs Weekend:")
    print(f"  Weekday: {wd_rate:.1%} late rate")
    print(f"  Weekend: {we_rate:.1%} late rate")
    print(f"  Difference: {diff * 100:.1f} percentage points")


if __name__ == "__main__":
    main()
