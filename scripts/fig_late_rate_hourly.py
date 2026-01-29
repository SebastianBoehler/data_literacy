#!/usr/bin/env python3
"""
Late Rate by Hour with Bootstrap Confidence Intervals

Compares mean delay vs late rate P(delay > threshold) by hour of day.
Late rate is more interpretable and robust to outliers for skewed delay data.

Output: outputs/late_rate_vs_mean_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "late_rate_vs_mean_comparison.png"

# Constants
LATE_THRESHOLD = 2  # minutes - common definition of 'late'
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def bootstrap_ci(data, func, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL):
    """Calculate bootstrap confidence interval for a statistic."""
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(func(sample))
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return np.mean(boot_stats), lower, upper


def main():
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df_valid = df.dropna(subset=['delay_minutes', 'hour'])
    print(f"Loaded {len(df_valid):,} records with delay and hour data")

    # Calculate stats by hour
    hourly_stats = []
    for hour in range(24):
        hour_data = df_valid[df_valid['hour'] == hour]['delay_minutes'].values
        if len(hour_data) > 10:
            # Late rate
            late_rate = (hour_data > LATE_THRESHOLD).mean()
            _, late_lower, late_upper = bootstrap_ci(
                hour_data, lambda x: (x > LATE_THRESHOLD).mean()
            )

            # Mean delay for comparison
            mean_delay = hour_data.mean()
            _, mean_lower, mean_upper = bootstrap_ci(hour_data, np.mean)

            hourly_stats.append({
                'hour': hour,
                'n': len(hour_data),
                'late_rate': late_rate,
                'late_lower': late_lower,
                'late_upper': late_upper,
                'mean_delay': mean_delay,
                'mean_lower': mean_lower,
                'mean_upper': mean_upper,
            })

    hourly_df = pd.DataFrame(hourly_stats)

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean delay with CI
    ax1 = axes[0]
    ax1.plot(hourly_df['hour'], hourly_df['mean_delay'], 'o-', color='steelblue', label='Mean delay')
    ax1.fill_between(
        hourly_df['hour'], hourly_df['mean_lower'], hourly_df['mean_upper'],
        alpha=0.3, color='steelblue', label='95% CI'
    )
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Mean Delay (minutes)')
    ax1.set_title('Mean Delay by Hour (with Bootstrap CI)')
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Late rate with CI
    ax2 = axes[1]
    ax2.plot(
        hourly_df['hour'], hourly_df['late_rate'] * 100, 'o-',
        color='darkred', label=f'P(delay > {LATE_THRESHOLD} min)'
    )
    ax2.fill_between(
        hourly_df['hour'], hourly_df['late_lower'] * 100, hourly_df['late_upper'] * 100,
        alpha=0.3, color='darkred', label='95% CI'
    )
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Late Rate (%)')
    ax2.set_title(f'Late Rate by Hour: P(delay > {LATE_THRESHOLD} min) with Bootstrap CI')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_ylim(0, 50)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add sample size annotation
    for ax in axes:
        ax.text(
            0.02, 0.98, f'n = {len(df_valid):,} total',
            transform=ax.transAxes, fontsize=9, va='top', ha='left', color='gray'
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Print insights
    delays = df_valid['delay_minutes']
    print(f'\nKey insights:')
    print(f'  Overall late rate P(delay > {LATE_THRESHOLD} min): {(delays > LATE_THRESHOLD).mean():.1%}')
    print(f'  Peak late rate: Hour {hourly_df.loc[hourly_df["late_rate"].idxmax(), "hour"]:.0f} '
          f'({hourly_df["late_rate"].max():.1%})')
    print(f'  Lowest late rate: Hour {hourly_df.loc[hourly_df["late_rate"].idxmin(), "hour"]:.0f} '
          f'({hourly_df["late_rate"].min():.1%})')


if __name__ == "__main__":
    main()
