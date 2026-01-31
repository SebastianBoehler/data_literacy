"""
Weather Effect on Late Rate with Bootstrap Confidence Intervals

Shows how temperature affects the probability of buses being late.
Includes sample size annotations and uncertainty visualization.

Output: outputs/weather_effect_with_ci.png
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

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "weather_effect_with_ci.png"

# Constants
LATE_THRESHOLD = 2  # minutes
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RANDOM_SEED = 42
MIN_SAMPLES = 50  # minimum samples per bin

# Temperature bins
TEMP_BINS = [(-10, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 35)]
TEMP_LABELS = ['<0°C', '0-5°C', '5-10°C', '10-15°C', '15-20°C', '20-25°C', '>25°C']


def bootstrap_ci(data, func, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL):
    """Calculate bootstrap confidence interval for a statistic."""
    n = len(data)
    boot_stats = [func(np.random.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df_valid = df.dropna(subset=['delay_minutes', 'temperature'])
    print(f"Loaded {len(df_valid):,} records with delay and temperature data")

    # Calculate stats by temperature bin
    weather_stats = []
    for (low, high), label in zip(TEMP_BINS, TEMP_LABELS):
        mask = (df_valid['temperature'] >= low) & (df_valid['temperature'] < high)
        temp_data = df_valid[mask]['delay_minutes'].values

        if len(temp_data) > MIN_SAMPLES:
            late_rate = (temp_data > LATE_THRESHOLD).mean()
            late_lower, late_upper = bootstrap_ci(
                temp_data, lambda x: (x > LATE_THRESHOLD).mean()
            )

            weather_stats.append({
                'temp_range': label,
                'n': len(temp_data),
                'late_rate': late_rate,
                'late_lower': late_lower,
                'late_upper': late_upper,
            })

    weather_df = pd.DataFrame(weather_stats)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(weather_df))
    ax.bar(x, weather_df['late_rate'] * 100, color='steelblue', alpha=0.7, edgecolor='black')

    # Add error bars for CI
    yerr_lower = (weather_df['late_rate'] - weather_df['late_lower']) * 100
    yerr_upper = (weather_df['late_upper'] - weather_df['late_rate']) * 100
    ax.errorbar(
        x, weather_df['late_rate'] * 100,
        yerr=[yerr_lower, yerr_upper],
        fmt='none', color='black', capsize=5, capthick=2, label='95% CI'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(weather_df['temp_range'])
    ax.set_xlabel('Temperature Range')
    ax.set_ylabel('Late Rate (%)')
    ax.set_title(f'Late Rate P(delay > {LATE_THRESHOLD} min) by Temperature with 95% Bootstrap CI')

    # Add sample sizes
    for i, (_, row) in enumerate(weather_df.iterrows()):
        ax.text(i, row['late_rate'] * 100 + 3, f'n={row["n"]:,}', ha='center', fontsize=8, color='gray')

    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Print summary
    print(f'\nWeather effect summary:')
    print(weather_df.to_string(index=False))

    # Key insight
    coldest = weather_df.iloc[0]
    warmest = weather_df[weather_df['n'] > 1000].iloc[-1]  # Exclude low-sample bins
    print(f'\nKey insight:')
    print(f'  Cold (<0°C): {coldest["late_rate"]:.1%} late rate')
    print(f'  Warmer ({warmest["temp_range"]}): {warmest["late_rate"]:.1%} late rate')


if __name__ == "__main__":
    main()
