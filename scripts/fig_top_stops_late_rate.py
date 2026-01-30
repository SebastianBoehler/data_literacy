#!/usr/bin/env python3
"""
Top Stops by Late Rate with Bootstrap CI and Mean Delay Color

Shows spatial heterogeneity - which stops have the highest probability of delays.
Includes 95% bootstrap confidence intervals, sample sizes, and mean delay as color.

Output: outputs/top_stops_late_rate.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
            mean_delay = stop_data.mean()
            ci_low, ci_high = bootstrap_ci(stop_data, lambda x: (x > LATE_THRESHOLD).mean())
            stop_stats.append({
                'stop': stop,
                'n': len(stop_data),
                'late_rate': late_rate,
                'mean_delay': mean_delay,
                'ci_low': ci_low,
                'ci_high': ci_high
            })

    stop_df = pd.DataFrame(stop_stats).sort_values('late_rate', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(stop_df))
    
    # Color bars by mean delay (using a colorblind-friendly colormap)
    cmap = plt.cm.YlOrBr  # Yellow-Orange-Brown (colorblind friendly)
    norm = Normalize(vmin=stop_df['mean_delay'].min(), vmax=stop_df['mean_delay'].max())
    colors = [cmap(norm(val)) for val in stop_df['mean_delay']]
    
    bars = ax.barh(y_pos, stop_df['late_rate'] * 100, color=colors, alpha=0.85, edgecolor='white')
    ax.errorbar(stop_df['late_rate'] * 100, y_pos,
                xerr=[(stop_df['late_rate'] - stop_df['ci_low']) * 100,
                      (stop_df['ci_high'] - stop_df['late_rate']) * 100],
                fmt='none', color='black', capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[:35] for s in stop_df['stop']])
    ax.set_xlabel('Late Rate (%)')
    ax.set_title(f'Top 15 Stops by Late Rate P(delay > {LATE_THRESHOLD} min)\nColor = Mean Delay (minutes)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add sample size and mean delay annotations
    for i, (_, row) in enumerate(stop_df.iterrows()):
        ax.text(row['late_rate'] * 100 + 1, i, 
                f'n={row["n"]:,} | μ={row["mean_delay"]:.1f}min', 
                va='center', fontsize=8, color='gray')

    # Add colorbar legend for mean delay
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Delay (minutes)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    print(f"\nTop 5 stops by late rate:")
    for _, row in stop_df.head(5).iterrows():
        print(f"  {row['stop']}: {row['late_rate']:.1%} (mean={row['mean_delay']:.1f}min) [{row['ci_low']:.1%}, {row['ci_high']:.1%}]")


if __name__ == "__main__":
    main()
