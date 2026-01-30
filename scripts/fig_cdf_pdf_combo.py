#!/usr/bin/env python3
"""
CDF + PDF Combo Plot for Delay Distribution

Creates a combined plot with:
- PDF (histogram as percentage/density) showing delay distribution
- CDF overlay allowing direct probability readings
- Key percentile markers for easy interpretation

This addresses feedback requesting:
- Percentage-based visualization instead of absolute counts
- CDF overlay for reading "X% have delay ≤ Y min"
- More interpretable than separate histogram

Output: docs/plots/{all,pre,post}/delay_cdf_pdf_combo.png
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
DOCS_PLOTS_DIR = SCRIPT_DIR / "docs" / "plots"

# Constants
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")
DELAY_RANGE = (-5, 20)
KEY_THRESHOLDS = [0, 2, 3, 5]  # Minutes to annotate


def create_cdf_pdf_combo(delays: np.ndarray, title: str, output_path: Path):
    """Create combined CDF + PDF plot with percentage readouts."""
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- PDF (Histogram as percentage) ---
    # Use density=False but normalize manually to get percentages
    counts, bins, patches = ax1.hist(
        delays, bins=50, range=DELAY_RANGE, 
        color='steelblue', alpha=0.6, edgecolor='white',
        label='Distribution'
    )
    
    # Convert to percentage
    total = len(delays)
    for patch, count in zip(patches, counts):
        patch.set_height(count / total * 100)
    
    ax1.set_ylim(0, max(counts) / total * 100 * 1.1)
    ax1.set_ylabel('Percentage of Buses (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # --- CDF (on secondary axis) ---
    ax2 = ax1.twinx()
    
    sorted_delays = np.sort(delays)
    n = len(sorted_delays)
    ecdf_y = np.arange(1, n + 1) / n
    
    # Subsample for smooth plotting
    step = max(1, n // 2000)
    ax2.plot(sorted_delays[::step], ecdf_y[::step] * 100, 
             color='darkred', linewidth=2.5, label='Cumulative %')
    
    ax2.set_ylabel('Cumulative Percentage (%)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0, 100)
    
    # --- Add key threshold annotations ---
    for threshold in KEY_THRESHOLDS:
        pct = (delays <= threshold).mean() * 100
        
        # Vertical line at threshold
        ax1.axvline(threshold, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Horizontal line from CDF to y-axis
        ax2.hlines(pct, DELAY_RANGE[0], threshold, colors='darkred', 
                   linestyles='--', alpha=0.4, linewidth=1)
        
        # Annotation
        label = 'On time' if threshold == 0 else f'≤{threshold} min'
        ax2.annotate(
            f'{label}: {pct:.1f}%',
            xy=(threshold, pct),
            xytext=(threshold + 1.5, pct + 3),
            fontsize=9,
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
            arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.6)
        )
    
    # --- Styling ---
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_title(title, fontweight='bold', fontsize=12)
    ax1.set_xlim(DELAY_RANGE)
    ax1.grid(axis='y', alpha=0.3)
    
    # Stats box
    stats_text = (
        f'n = {len(delays):,}\n'
        f'Mean = {np.mean(delays):.2f} min\n'
        f'Median = {np.median(delays):.2f} min\n'
        f'Late (>2 min) = {(delays > 2).mean():.1%}'
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    delays_all = df['delay_minutes'].dropna().values
    
    # Split by schedule change
    delays_pre = df[df['timestamp'] < SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna().values
    delays_post = df[df['timestamp'] >= SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna().values
    
    print(f"Total: {len(delays_all):,}, Pre: {len(delays_pre):,}, Post: {len(delays_post):,}")
    
    # Generate for each period
    periods = [
        ('all', delays_all, 'Delay Distribution with CDF — All Data'),
        ('pre', delays_pre, 'Delay Distribution with CDF — Before Schedule Change'),
        ('post', delays_post, 'Delay Distribution with CDF — After Schedule Change'),
    ]
    
    for period, delays, title in periods:
        output_dir = DOCS_PLOTS_DIR / period
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "delay_cdf_pdf_combo.png"
        create_cdf_pdf_combo(delays, title, output_path)
    
    # Also save to outputs for easy access
    output_path = SCRIPT_DIR / "outputs" / "delay_cdf_pdf_combo.png"
    create_cdf_pdf_combo(delays_all, 'Delay Distribution with CDF — All Data', output_path)
    
    print("\nDone! Generated CDF+PDF combo plots for all periods.")


if __name__ == "__main__":
    main()
