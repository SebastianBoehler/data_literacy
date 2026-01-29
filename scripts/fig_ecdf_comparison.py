#!/usr/bin/env python3
"""
ECDF vs Histogram Comparison Plot

Demonstrates the advantage of ECDFs over histograms for delay distribution:
- ECDFs allow direct probability readings
- No arbitrary bin width choices
- DKW confidence bands show uncertainty

Output: outputs/ecdf_vs_histogram_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "ecdf_vs_histogram_comparison.png"

# Constants
DELAY_RANGE = (-10, 30)  # minutes
DKW_ALPHA = 0.05  # 95% confidence


def main():
    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    delays = df['delay_minutes'].dropna()
    print(f"Loaded {len(delays):,} delay records")

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram (traditional approach)
    ax1 = axes[0]
    ax1.hist(delays, bins=50, range=DELAY_RANGE, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', label='On time')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram: Delay Distribution')
    ax1.legend()

    # Right: ECDF (recommended approach)
    ax2 = axes[1]
    sorted_delays = np.sort(delays)
    ecdf_y = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

    # Subsample for plotting (too many points)
    step = max(1, len(sorted_delays) // 5000)
    ax2.plot(sorted_delays[::step], ecdf_y[::step], color='steelblue', linewidth=1.5)
    ax2.axvline(0, color='red', linestyle='--', label='On time')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Median')

    # Add DKW confidence band (95%)
    n = len(delays)
    epsilon = np.sqrt(np.log(2 / DKW_ALPHA) / (2 * n))
    ax2.fill_between(
        sorted_delays[::step],
        np.clip(ecdf_y[::step] - epsilon, 0, 1),
        np.clip(ecdf_y[::step] + epsilon, 0, 1),
        alpha=0.2, color='steelblue', label='95% DKW band'
    )

    ax2.set_xlabel('Delay (minutes)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('ECDF: Delay Distribution with Uncertainty')
    ax2.set_xlim(DELAY_RANGE)
    ax2.legend()

    # Add annotations showing what ECDF tells you
    on_time_rate = (delays <= 0).mean()
    median_delay = delays.median()

    ax2.annotate(
        f'P(delay ≤ 0) ≈ {on_time_rate:.1%}',
        xy=(0, on_time_rate), xytext=(5, 0.3),
        arrowprops=dict(arrowstyle='->', color='darkred'),
        fontsize=10, color='darkred'
    )

    ax2.annotate(
        f'Median ≈ {median_delay:.1f} min',
        xy=(median_delay, 0.5), xytext=(15, 0.55),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10, color='gray'
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Print key stats
    print(f'\nKey insights from ECDF:')
    print(f'  P(delay ≤ 0) = {on_time_rate:.1%} (on time or early)')
    print(f'  P(delay > 2 min) = {(delays > 2).mean():.1%} (late)')
    print(f'  P(delay > 5 min) = {(delays > 5).mean():.1%} (significantly late)')
    print(f'  Median delay = {median_delay:.1f} min')
    print(f'  Mean delay = {delays.mean():.1f} min')


if __name__ == "__main__":
    main()
