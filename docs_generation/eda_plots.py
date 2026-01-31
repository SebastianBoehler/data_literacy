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
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    print("Using tueplots styling")
except ImportError:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
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


def generate_cdf_pdf_combo_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate combined CDF + PDF plot with percentage readouts.
    
    Addresses feedback requesting:
    - Percentage-based visualization instead of absolute counts
    - CDF overlay for reading "X% have delay ≤ Y min"
    """
    delays = df['delay_minutes'].dropna().values
    delay_range = (-5, 20)
    key_thresholds = [0, 2, 3, 5]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- PDF (Histogram as percentage) ---
    counts, bins, patches = ax1.hist(
        delays, bins=50, range=delay_range, 
        color='steelblue', alpha=0.6, edgecolor='white',
        label='Distribution'
    )
    
    # Convert to percentage
    total = len(delays)
    for patch, count in zip(patches, counts):
        patch.set_height(count / total * 100)
    
    ax1.set_ylim(0, 100)
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
    for threshold in key_thresholds:
        pct = (delays <= threshold).mean() * 100
        
        # Vertical line at threshold
        ax1.axvline(threshold, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Horizontal line from CDF to y-axis
        ax2.hlines(pct, delay_range[0], threshold, colors='darkred', 
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
    ax1.set_title(f'Delay Distribution with CDF ({period_label})', fontweight='bold', fontsize=12)
    ax1.set_xlim(delay_range)
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
    output_path = output_dir / "delay_cdf_pdf_combo.png"
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
    """Generate top delayed lines bar chart with uniform coloring (no misleading gradient)."""
    
    line_stats = df.groupby('line_name')['delay_minutes'].agg(['mean', 'count']).reset_index()
    line_stats = line_stats[line_stats['count'] >= 100]  # Filter small samples
    line_stats = line_stats.sort_values('mean', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use uniform color - no misleading red-green gradient
    bars = ax.barh(line_stats['line_name'], line_stats['mean'], color='steelblue', edgecolor='white')
    
    # Add value labels at end of bars
    for bar, val in zip(bars, line_stats['mean']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=8)
    
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
    
    # Use uniform color - no misleading red-green gradient
    bars = ax.barh(line_stats['line_name'], line_stats['mean'], color='steelblue', edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, line_stats['mean']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=7)
    
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


def generate_late_rate_hourly(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate late rate by hour with bootstrap CI."""
    
    LATE_THRESHOLD = 2  # minutes
    N_BOOTSTRAP = 500
    np.random.seed(42)
    
    df_valid = df.dropna(subset=['delay_minutes'])
    df_valid = df_valid.copy()
    df_valid['hour'] = pd.to_datetime(df_valid['timestamp']).dt.hour
    
    hourly_stats = []
    for hour in range(24):
        hour_data = df_valid[df_valid['hour'] == hour]['delay_minutes'].values
        if len(hour_data) > 10:
            late_rate = (hour_data > LATE_THRESHOLD).mean()
            # Bootstrap CI
            boot_rates = [np.mean(np.random.choice(hour_data, len(hour_data), replace=True) > LATE_THRESHOLD) 
                          for _ in range(N_BOOTSTRAP)]
            hourly_stats.append({
                'hour': hour,
                'late_rate': late_rate,
                'ci_lower': np.percentile(boot_rates, 2.5),
                'ci_upper': np.percentile(boot_rates, 97.5),
            })
    
    hourly_df = pd.DataFrame(hourly_stats)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_df['hour'], hourly_df['late_rate'] * 100, 'o-', color='darkred', linewidth=2)
    ax.fill_between(hourly_df['hour'], hourly_df['ci_lower'] * 100, hourly_df['ci_upper'] * 100,
                    alpha=0.3, color='darkred', label='95% Bootstrap CI')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Late Rate (%)')
    ax.set_title(f'Late Rate P(delay > {LATE_THRESHOLD} min) by Hour ({period_label})', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    # Auto-scale Y-axis to fit all data with padding
    y_max = max(hourly_df['ci_upper'].max() * 100, hourly_df['late_rate'].max() * 100) * 1.1
    ax.set_ylim(0, max(y_max, 50))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.98, f'n = {len(df_valid):,}', transform=ax.transAxes, fontsize=9, va='top', color='gray')
    
    plt.tight_layout()
    output_path = output_dir / "late_rate_hourly.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_weather_effect(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate weather effect on late rate with bootstrap CI."""
    
    LATE_THRESHOLD = 2
    N_BOOTSTRAP = 500
    MIN_SAMPLES = 50
    np.random.seed(42)
    
    df_valid = df.dropna(subset=['delay_minutes', 'temperature'])
    
    temp_bins = [(-10, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
    temp_labels = ['<0°C', '0-5°C', '5-10°C', '10-15°C', '15-20°C']
    
    weather_stats = []
    for (low, high), label in zip(temp_bins, temp_labels):
        mask = (df_valid['temperature'] >= low) & (df_valid['temperature'] < high)
        temp_data = df_valid[mask]['delay_minutes'].values
        
        if len(temp_data) > MIN_SAMPLES:
            late_rate = (temp_data > LATE_THRESHOLD).mean()
            boot_rates = [np.mean(np.random.choice(temp_data, len(temp_data), replace=True) > LATE_THRESHOLD)
                          for _ in range(N_BOOTSTRAP)]
            weather_stats.append({
                'temp_range': label,
                'n': len(temp_data),
                'late_rate': late_rate,
                'ci_lower': np.percentile(boot_rates, 2.5),
                'ci_upper': np.percentile(boot_rates, 97.5),
            })
    
    if not weather_stats:
        print(f"  Skipped weather_effect.png (no data)")
        return
    
    weather_df = pd.DataFrame(weather_stats)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(weather_df))
    ax.bar(x, weather_df['late_rate'] * 100, color='steelblue', alpha=0.7, edgecolor='black')
    
    yerr_lower = (weather_df['late_rate'] - weather_df['ci_lower']) * 100
    yerr_upper = (weather_df['ci_upper'] - weather_df['late_rate']) * 100
    ax.errorbar(x, weather_df['late_rate'] * 100, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=5, capthick=2, label='95% CI')
    
    ax.set_xticks(x)
    ax.set_xticklabels(weather_df['temp_range'])
    ax.set_xlabel('Temperature Range')
    ax.set_ylabel('Late Rate (%)')
    ax.set_title(f'Late Rate by Temperature ({period_label})', fontweight='bold')
    
    for i, row in weather_df.iterrows():
        ax.text(i, row['late_rate'] * 100 + 2, f'n={row["n"]:,}', ha='center', fontsize=8, color='gray')
    
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / "weather_effect.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_delay_accumulation_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate delay accumulation along route plot.
    
    Shows how delays increase as buses progress through their routes,
    using stop_sequence as a proxy for route position.
    """
    
    # Filter valid data
    valid = df.dropna(subset=['stop_sequence', 'delay_minutes'])
    valid = valid[valid['stop_sequence'] >= 1]
    
    if len(valid) < 100:
        print(f"  Skipped delay_accumulation.png (insufficient data)")
        return
    
    # Calculate stats by stop sequence
    seq_stats = valid.groupby('stop_sequence')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    seq_stats = seq_stats[seq_stats['count'] >= 50]  # filter small samples
    seq_stats['ci95'] = 1.96 * seq_stats['std'] / np.sqrt(seq_stats['count'])
    
    # Calculate correlation
    corr = valid['stop_sequence'].corr(valid['delay_minutes'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel A: Mean delay by stop position ---
    ax = axes[0]
    ax.fill_between(seq_stats['stop_sequence'], 
                    seq_stats['mean'] - seq_stats['ci95'],
                    seq_stats['mean'] + seq_stats['ci95'],
                    alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(seq_stats['stop_sequence'], seq_stats['mean'], 'o-', 
            color='steelblue', linewidth=2, markersize=5, label='Mean delay')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Stop Position in Route')
    ax.set_ylabel('Mean Delay (minutes)')
    ax.set_title('(A) Delay Accumulation Along Route', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add correlation annotation
    ax.text(0.98, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Panel B: Scatter with regression line ---
    ax = axes[1]
    sample = valid.sample(min(5000, len(valid)), random_state=42)
    ax.scatter(sample['stop_sequence'], sample['delay_minutes'],
               alpha=0.1, s=8, color='steelblue')
    
    # Add regression line
    z = np.polyfit(valid['stop_sequence'], valid['delay_minutes'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, seq_stats['stop_sequence'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2.5, label=f'Linear fit (r = {corr:.3f})')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Stop Position in Route')
    ax.set_ylabel('Delay (minutes)')
    ax.set_title('(B) Delay vs Stop Position (Sample)', fontweight='bold')
    ax.set_ylim(-5, 30)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add stats annotation
    stats_text = (f'n = {len(valid):,}\n'
                  f'Slope = {z[0]:.3f} min/stop\n'
                  f'Mean at stop 1: {seq_stats[seq_stats["stop_sequence"]==1]["mean"].values[0]:.2f} min' 
                  if 1 in seq_stats['stop_sequence'].values else f'n = {len(valid):,}')
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add overall title
    fig.suptitle(f'Delay Accumulation Analysis — {period_label}', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / "delay_accumulation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_schedule_change_ecdf(df_pre: pd.DataFrame, df_post: pd.DataFrame, output_dir: Path):
    """Generate pre/post schedule change ECDF comparison."""
    
    LATE_THRESHOLD = 2
    DKW_ALPHA = 0.05
    
    pre = df_pre['delay_minutes'].dropna()
    post = df_post['delay_minutes'].dropna()
    
    if len(pre) < 100 or len(post) < 100:
        print(f"  Skipped schedule_change_ecdf.png (insufficient data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pre ECDF
    pre_sorted = np.sort(pre)
    pre_ecdf = np.arange(1, len(pre_sorted) + 1) / len(pre_sorted)
    step_pre = max(1, len(pre_sorted) // 2000)
    
    # Post ECDF
    post_sorted = np.sort(post)
    post_ecdf = np.arange(1, len(post_sorted) + 1) / len(post_sorted)
    step_post = max(1, len(post_sorted) // 2000)
    
    ax.plot(pre_sorted[::step_pre], pre_ecdf[::step_pre], color='steelblue', linewidth=2, 
            label=f'Before (n={len(pre):,})')
    ax.plot(post_sorted[::step_post], post_ecdf[::step_post], color='darkorange', linewidth=2,
            label=f'After (n={len(post):,})')
    
    # DKW bands
    eps_pre = np.sqrt(np.log(2/DKW_ALPHA) / (2*len(pre)))
    eps_post = np.sqrt(np.log(2/DKW_ALPHA) / (2*len(post)))
    
    ax.fill_between(pre_sorted[::step_pre], 
                    np.clip(pre_ecdf[::step_pre] - eps_pre, 0, 1),
                    np.clip(pre_ecdf[::step_pre] + eps_pre, 0, 1),
                    alpha=0.15, color='steelblue')
    ax.fill_between(post_sorted[::step_post],
                    np.clip(post_ecdf[::step_post] - eps_post, 0, 1),
                    np.clip(post_ecdf[::step_post] + eps_post, 0, 1),
                    alpha=0.15, color='darkorange')
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(LATE_THRESHOLD, color='red', linestyle=':', alpha=0.7, label=f'Late ({LATE_THRESHOLD} min)')
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Delay Distribution: Before vs After Schedule Change', fontweight='bold')
    ax.set_xlim(-5, 20)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "schedule_change_ecdf.png"
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
        
        # Generate additional uncertainty-aware plots
        generate_late_rate_hourly(period_df, period, config['label'], output_dir)
        generate_weather_effect(period_df, period, config['label'], output_dir)
        generate_delay_accumulation_plot(period_df, period, config['label'], output_dir)
        generate_cdf_pdf_combo_plot(period_df, period, config['label'], output_dir)
    
    # Generate schedule change comparison (only once, in 'all' directory)
    all_output_dir = PLOTS_DIR / "all"
    all_output_dir.mkdir(parents=True, exist_ok=True)
    generate_schedule_change_ecdf(df_pre, df_post, all_output_dir)
    
    print("\n" + "=" * 70)
    print(f"Done! Generated plots for {len(PERIODS)} periods in docs/plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
