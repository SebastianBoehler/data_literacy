#!/usr/bin/env python3
"""
Generate EDA plots for GitHub Pages visualization.
Generates plots for three time periods: all, pre, post.

Usage:
1. Run the notebook to export all_trip_data.parquet
2. Run this script: python scripts/generate_eda_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to use tueplots if available
try:
    from tueplots import bundles, axes
    plt.rcParams.update({
        **bundles.icml2024(usetex=False, family="serif"),
        **axes.lines(),
        "figure.dpi": 150,
    })
    print("Using tueplots styling")
except ImportError:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi': 150,
    })
    print("tueplots not available, using default styling")

# Paths
SCRIPT_DIR = Path(__file__).parent.parent  # docs_generation/ -> code/
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
DOCS_DIR = SCRIPT_DIR / "docs"
PLOTS_DIR = DOCS_DIR / "plots"

# Schedule change date
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")

# Period configurations
PERIODS = {
    "all": {"label": "All Data", "filter": None},
    "pre": {"label": "Before Schedule Change", "filter": "pre"},
    "post": {"label": "After Schedule Change", "filter": "post"},
}


def load_data():
    """Load trip data from parquet file."""
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        return None
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} records")
    return df


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter dataframe by time period."""
    if period == "pre":
        return df[df['timestamp'] < SCHEDULE_CHANGE_DATE].copy()
    elif period == "post":
        return df[df['timestamp'] >= SCHEDULE_CHANGE_DATE].copy()
    return df.copy()


def generate_delay_distribution_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate delay distribution histogram."""
    
    delays = df['delay_minutes'].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    ax.hist(delays, bins=50, range=(-5, 20), color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='darkred', linestyle='--', linewidth=1.5, label='On Time')
    ax.axvline(delays.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'Mean: {delays.mean():.2f} min')
    ax.axvline(delays.median(), color='green', linestyle='-', linewidth=1.5, label=f'Median: {delays.median():.2f} min')
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Delay Distribution ({period_label})', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Stats text
    stats_text = f'n = {len(delays):,}\nStd = {delays.std():.2f} min'
    ax.text(0.98, 0.75, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = output_dir / "delay_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_hourly_delay_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate hourly delay pattern plot."""
    
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    hourly_stats = df.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    hourly_stats['ci95'] = 1.96 * hourly_stats['std'] / np.sqrt(hourly_stats['count'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(hourly_stats['hour'], 
                    hourly_stats['mean'] - hourly_stats['ci95'],
                    hourly_stats['mean'] + hourly_stats['ci95'],
                    alpha=0.3, color='steelblue')
    ax.plot(hourly_stats['hour'], hourly_stats['mean'], 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.axhline(0, color='darkred', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Delay (minutes)')
    ax.set_title(f'Delay by Hour of Day ({period_label})', fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "hourly_delay.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_delay_ecdf_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate ECDF plot with DKW confidence bands."""
    
    delays = df['delay_minutes'].dropna().values
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Panel A: ECDF ---
    ax = axes[0]
    x_sorted = np.sort(delays)
    n = len(x_sorted)
    ecdf_y = np.arange(1, n + 1) / n
    
    # DKW confidence bands (95%)
    alpha = 0.05
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.clip(ecdf_y - epsilon, 0, 1)
    upper = np.clip(ecdf_y + epsilon, 0, 1)
    
    ax.fill_between(x_sorted, lower, upper, alpha=0.3, color='steelblue', label='95% DKW Band')
    ax.step(x_sorted, ecdf_y, where='post', color='steelblue', linewidth=1.5, label='ECDF')
    ax.axvline(0, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8, label='On Time (0 min)')
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'(A) Empirical CDF ({period_label})', fontweight='bold')
    ax.set_xlim(-5, 15)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # --- Panel B: Punctuality Distribution ---
    ax = axes[1]
    late_rate = np.mean(delays > 0)
    on_time_rate = np.mean(delays == 0)
    early_rate = np.mean(delays < 0)
    
    categories = ['Early\n(< 0 min)', 'On Time\n(= 0 min)', 'Late\n(> 0 min)']
    values = [early_rate * 100, on_time_rate * 100, late_rate * 100]
    colors = ['forestgreen', 'steelblue', 'firebrick']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage of Departures')
    ax.set_title(f'(B) Punctuality Distribution ({period_label})', fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Stats text
    stats_text = (f'n = {n:,}\n'
                  f'Mean = {np.mean(delays):.2f} min\n'
                  f'Median = {np.median(delays):.2f} min')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = output_dir / "delay_ecdf.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_top_delayed_lines_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate top delayed lines bar chart with gradient coloring."""
    
    line_stats = df.groupby('line_name')['delay_minutes'].agg(['mean', 'count']).reset_index()
    line_stats = line_stats[line_stats['count'] >= 100]  # Filter small samples
    line_stats = line_stats.sort_values('mean', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use gradient coloring based on delay values (green -> yellow -> red)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('delay_gradient', ['#2ecc71', '#f1c40f', '#e74c3c'])
    
    # Normalize delays to 0-1 range for colormap
    delay_values = line_stats['mean'].values
    norm = plt.Normalize(vmin=delay_values.min(), vmax=delay_values.max())
    colors = [cmap(norm(val)) for val in delay_values]
    
    bars = ax.barh(line_stats['line_name'], line_stats['mean'], color=colors, edgecolor='white')
    
    ax.axvline(0, color='darkred', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Mean Delay (minutes)')
    ax.set_ylabel('Line')
    ax.set_title(f'Top 15 Lines by Mean Delay ({period_label})', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "top_delayed_lines.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_combined_eda_figure(df: pd.DataFrame, period: str, period_label: str, output_dir: Path,
                                  df_pre: pd.DataFrame = None, df_post: pd.DataFrame = None):
    """Generate combined 2x2 EDA figure with subplots A, B, C, D.
    
    For panels A and B, shows pre/post schedule change comparison with overlays.
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Panel A: Delay Distribution Histogram ---
    ax = axes[0, 0]
    delays = df['delay_minutes'].dropna()
    
    # Main histogram (current period data only)
    ax.hist(delays, bins=50, range=(-1, 20), color='steelblue', edgecolor='white', alpha=0.8)
    
    ax.axvline(delays.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'Mean: {delays.mean():.2f} min')
    ax.axvline(delays.median(), color='green', linestyle='-', linewidth=1.5, label=f'Median: {delays.median():.2f} min')
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('(A) Delay Distribution', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_xlim(-1, 20)
    
    stats_text = f'n = {len(delays):,}\nStd = {delays.std():.2f} min'
    ax.text(0.98, 0.75, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Panel B: Hourly Delay Pattern with pre/post comparison ---
    ax = axes[0, 1]
    df_hourly = df.copy()
    df_hourly['hour'] = pd.to_datetime(df_hourly['timestamp']).dt.hour
    
    hourly_stats = df_hourly.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    hourly_stats['ci95'] = 1.96 * hourly_stats['std'] / np.sqrt(hourly_stats['count'])
    
    # Overlay pre/post if available
    if df_pre is not None and df_post is not None:
        df_pre_h = df_pre.copy()
        df_pre_h['hour'] = pd.to_datetime(df_pre_h['timestamp']).dt.hour
        pre_stats = df_pre_h.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
        pre_stats['ci95'] = 1.96 * pre_stats['std'] / np.sqrt(pre_stats['count'])
        
        df_post_h = df_post.copy()
        df_post_h['hour'] = pd.to_datetime(df_post_h['timestamp']).dt.hour
        post_stats = df_post_h.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
        post_stats['ci95'] = 1.96 * post_stats['std'] / np.sqrt(post_stats['count'])
        
        # Pre schedule change (red, lower opacity)
        ax.fill_between(pre_stats['hour'], 
                        pre_stats['mean'] - pre_stats['ci95'],
                        pre_stats['mean'] + pre_stats['ci95'],
                        alpha=0.15, color='#e74c3c')
        ax.plot(pre_stats['hour'], pre_stats['mean'], 'o--', color='#e74c3c', linewidth=1.5, markersize=4, alpha=0.7, label='Pre')
        
        # Post schedule change (green, lower opacity)
        ax.fill_between(post_stats['hour'], 
                        post_stats['mean'] - post_stats['ci95'],
                        post_stats['mean'] + post_stats['ci95'],
                        alpha=0.15, color='#2ecc71')
        ax.plot(post_stats['hour'], post_stats['mean'], 'o--', color='#2ecc71', linewidth=1.5, markersize=4, alpha=0.7, label='Post')
    
    # Main line (all data) - on top
    ax.fill_between(hourly_stats['hour'], 
                    hourly_stats['mean'] - hourly_stats['ci95'],
                    hourly_stats['mean'] + hourly_stats['ci95'],
                    alpha=0.3, color='steelblue')
    ax.plot(hourly_stats['hour'], hourly_stats['mean'], 'o-', color='steelblue', linewidth=2, markersize=5, label='All')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Delay (minutes)')
    ax.set_title('(B) Delay by Hour of Day', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    # --- Panel C: ECDF ---
    ax = axes[1, 0]
    delay_vals = delays.values
    x_sorted = np.sort(delay_vals)
    n = len(x_sorted)
    ecdf_y = np.arange(1, n + 1) / n
    
    # DKW confidence bands (95%)
    alpha_dkw = 0.05
    epsilon = np.sqrt(np.log(2 / alpha_dkw) / (2 * n))
    lower = np.clip(ecdf_y - epsilon, 0, 1)
    upper = np.clip(ecdf_y + epsilon, 0, 1)
    
    ax.fill_between(x_sorted, lower, upper, alpha=0.3, color='steelblue', label='95% DKW Band')
    ax.step(x_sorted, ecdf_y, where='post', color='steelblue', linewidth=1.5, label='ECDF')
    ax.axvline(0, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8, label='On Time')
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('(C) Empirical CDF', fontweight='bold')
    ax.set_xlim(-5, 15)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # --- Panel D: Top 15 Delayed Lines ---
    ax = axes[1, 1]
    line_stats = df.groupby('line_name')['delay_minutes'].agg(['mean', 'count']).reset_index()
    line_stats = line_stats[line_stats['count'] >= 100]
    line_stats = line_stats.sort_values('mean', ascending=False).head(15)
    
    cmap = LinearSegmentedColormap.from_list('delay_gradient', ['#2ecc71', '#f1c40f', '#e74c3c'])
    delay_values = line_stats['mean'].values
    norm = plt.Normalize(vmin=delay_values.min(), vmax=delay_values.max())
    colors = [cmap(norm(val)) for val in delay_values]
    
    ax.barh(line_stats['line_name'], line_stats['mean'], color=colors, edgecolor='white')
    ax.axvline(0, color='darkred', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Mean Delay (minutes)')
    ax.set_ylabel('Line')
    ax.set_title('(D) Top 15 Lines by Mean Delay', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Delay Analysis — {period_label}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / "eda_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("Generating EDA Plots for GitHub Pages")
    print("(with period filtering: all, pre, post)")
    print("=" * 70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Pre-compute pre/post dataframes for comparison overlays
    df_pre = filter_by_period(df, 'pre')
    df_post = filter_by_period(df, 'post')
    print(f"Pre-schedule records: {len(df_pre):,}")
    print(f"Post-schedule records: {len(df_post):,}")
    
    # Generate plots for each period
    for period, config in PERIODS.items():
        print(f"\n{'='*50}")
        print(f"Period: {period} ({config['label']})")
        print(f"{'='*50}")
        
        # Create period-specific directory
        output_dir = PLOTS_DIR / period
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter data
        period_df = filter_by_period(df, period)
        print(f"  Records: {len(period_df):,}")
        
        if period_df.empty:
            print(f"  WARNING: No data for period '{period}'")
            continue
        
        # Generate combined figure with all 4 subplots
        # Pass pre/post data for comparison overlays in panels A and B
        generate_combined_eda_figure(period_df, period, config['label'], output_dir,
                                     df_pre=df_pre, df_post=df_post)
    
    print("\n" + "=" * 70)
    print(f"Done! Generated plots for {len(PERIODS)} periods in docs/plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
