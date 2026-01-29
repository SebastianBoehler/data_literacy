#!/usr/bin/env python3
"""
Top Stops by Late Rate with Bootstrap CI

Shows spatial heterogeneity - which stops have the highest probability of delays.
Includes 95% bootstrap confidence intervals and sample sizes.

Output: outputs/top_stops_late_rate.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "top_stops_late_rate.png"

LATE_THRESHOLD = 2
N_BOOTSTRAP = 500
MIN_SAMPLES = 100
RANDOM_SEED = 42


def bootstrap_ci(data, func, n_boot=N_BOOTSTRAP):
    stats = [func(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def main():
    np.random.seed(RANDOM_SEED)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    stop_stats = []
    for stop in df['stop_name'].unique():
        stop_data = df[df['stop_name'] == stop]['delay_minutes'].dropna().values
        if len(stop_data) >= MIN_SAMPLES:
            late_rate = (stop_data > LATE_THRESHOLD).mean()
            ci_low, ci_high = bootstrap_ci(stop_data, lambda x: (x > LATE_THRESHOLD).mean())
            stop_stats.append({
                'stop': stop,
                'n': len(stop_data),
                'late_rate': late_rate,
                'ci_low': ci_low,
                'ci_high': ci_high
            })

    stop_df = pd.DataFrame(stop_stats).sort_values('late_rate', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(stop_df))
    ax.barh(y_pos, stop_df['late_rate'] * 100, color='steelblue', alpha=0.7)
    ax.errorbar(stop_df['late_rate'] * 100, y_pos,
                xerr=[(stop_df['late_rate'] - stop_df['ci_low']) * 100,
                      (stop_df['ci_high'] - stop_df['late_rate']) * 100],
                fmt='none', color='black', capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[:30] for s in stop_df['stop']])
    ax.set_xlabel('Late Rate (%)')
    ax.set_title(f'Top 15 Stops by Late Rate P(delay > {LATE_THRESHOLD} min) with 95% CI', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(stop_df.iterrows()):
        ax.text(row['late_rate'] * 100 + 1, i, f'n={row["n"]:,}', va='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    print(f"\nTop 5 stops by late rate:")
    for _, row in stop_df.head(5).iterrows():
        print(f"  {row['stop']}: {row['late_rate']:.1%} [{row['ci_low']:.1%}, {row['ci_high']:.1%}]")


if __name__ == "__main__":
    main()
