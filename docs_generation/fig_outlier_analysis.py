"""
Outlier Analysis - Extreme Delay Documentation

Documents extreme delay events (>30 min) to understand data quality.
Shows which dates and lines had major disruptions.

Output: outputs/outlier_analysis.png
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
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "outlier_analysis.png"

EXTREME_THRESHOLD = 30  # minutes


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Find extreme delays
    extreme = df[df['delay_minutes'] > EXTREME_THRESHOLD].copy()
    extreme['date'] = pd.to_datetime(extreme['timestamp']).dt.date

    # Count by date
    outlier_by_date = extreme.groupby('date').agg(
        count=('delay_minutes', 'count'),
        max_delay=('delay_minutes', 'max'),
        lines=('line_name', lambda x: ', '.join(sorted(set(x))))
    ).reset_index()
    outlier_by_date = outlier_by_date.sort_values('count', ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Extreme delays by date
    ax1 = axes[0]
    ax1.bar(range(len(outlier_by_date)), outlier_by_date['count'], color='darkred', alpha=0.7)
    ax1.set_xticks(range(len(outlier_by_date)))
    ax1.set_xticklabels([str(d) for d in outlier_by_date['date']], rotation=45, ha='right')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Number of Records with Delay > {EXTREME_THRESHOLD} min')
    ax1.set_title(f'Top 10 Days with Extreme Delays (>{EXTREME_THRESHOLD} min)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Distribution of extreme delays
    ax2 = axes[1]
    extreme_delays = df[df['delay_minutes'] > EXTREME_THRESHOLD]['delay_minutes'].values
    ax2.hist(extreme_delays, bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax2.axvline(extreme_delays.mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {extreme_delays.mean():.1f} min')
    ax2.axvline(np.median(extreme_delays), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(extreme_delays):.1f} min')
    ax2.set_xlabel('Delay (minutes)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of Extreme Delays (n={len(extreme_delays):,} records)', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Summary
    total = len(df)
    extreme_count = len(extreme)
    print(f'\nOutlier Summary:')
    print(f'  Total records: {total:,}')
    print(f'  Extreme delays (>{EXTREME_THRESHOLD} min): {extreme_count:,} ({extreme_count / total * 100:.2f}%)')
    print(f'\nTop 3 disruption days:')
    for _, row in outlier_by_date.head(3).iterrows():
        print(f'  {row["date"]}: {row["count"]} records, max {row["max_delay"]:.0f} min, lines: {row["lines"]}')


if __name__ == "__main__":
    main()
