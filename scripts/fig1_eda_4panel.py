"""
Figure 1: Exploratory Analysis of Tübingen Bus Delays (4-Panel)

Generates a 2x2 figure with:
(A) Delay Distribution with Histogram + CDF
(B) Top 10 Most Delayed Lines by average delay
(C) Top 10 Busiest Stops by departure count
(D) Mean delay by weather condition with sample sizes

Outputs:
- plots/fig1_eda_4panel.png
- plots/fig1_eda_4panel.pdf (for paper)
- paper/images/fig1_eda_4panel.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent  # scripts/ -> code/
sys.path.insert(0, str(SCRIPT_DIR))
from modules.plot_config import apply_style, STYLE

apply_style()

DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
PLOT_DIR = SCRIPT_DIR / "plots"
PAPER_DIR = SCRIPT_DIR / "paper" / "images"

PLOT_DIR.mkdir(exist_ok=True)
PAPER_DIR.mkdir(exist_ok=True)

# Consistent color for all panels
MAIN_BLUE = '#6baed6'
CDF_COLOR = '#8b0000'  # Dark red for CDF line


def load_data():
    """Load trip data from parquet file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} records")
    return df


def main():
    print("=" * 60)
    print("FIGURE 1: EDA 4-Panel")
    print("=" * 60)
    
    df = load_data()
    
    # Ensure delay_minutes column exists
    if 'delay_minutes' not in df.columns:
        if 'departure_delay_minutes' in df.columns:
            df['delay_minutes'] = df['departure_delay_minutes']
        elif 'arrival_delay_minutes' in df.columns:
            df['delay_minutes'] = df['arrival_delay_minutes']
    
    delays = df['delay_minutes'].dropna()
    print(f"Valid delay records: {len(delays):,}")
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    # =========================================================================
    # Panel A: Delay Distribution with Histogram + CDF (like docs)
    # =========================================================================
    ax = axes[0, 0]
    ax2 = ax.twinx()  # Secondary y-axis for CDF
    
    # Filter delays for display range
    delay_range = (-2, 20)
    delays_filtered = delays[(delays >= delay_range[0]) & (delays <= delay_range[1])]
    
    # Histogram as percentage per bin
    bins = np.arange(delay_range[0], delay_range[1] + 1, 1)
    counts, bin_edges = np.histogram(delays_filtered, bins=bins)
    percentages = counts / len(delays) * 100
    
    ax.bar(bin_edges[:-1] + 0.5, percentages, width=0.9, color=MAIN_BLUE, 
           edgecolor='white', alpha=0.8, label='% per bin')
    
    # CDF calculation on full data
    sorted_delays = np.sort(delays)
    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays) * 100
    
    # Plot CDF (filter to display range)
    mask = (sorted_delays >= delay_range[0]) & (sorted_delays <= delay_range[1])
    ax2.plot(sorted_delays[mask], cdf[mask], '-', color=CDF_COLOR, 
             linewidth=2, marker='o', markersize=2, label='Cumulative %')
    
    # Add key percentile annotations on CDF
    for threshold, label_offset in [(0, (0.5, -8)), (2, (0.5, -5)), (3, (0.5, -5)), (5, (0.5, -5))]:
        pct = (delays <= threshold).mean() * 100
        ax2.axhline(pct, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        if threshold == 0:
            ax2.annotate(
                f'On time (≤0): {pct:.1f}%',
                xy=(threshold, pct),
                xytext=(threshold + label_offset[0], pct + label_offset[1]),
                fontsize=8,
                color=CDF_COLOR,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'),
                arrowprops=dict(arrowstyle='->', color=CDF_COLOR, alpha=0.6, linewidth=0.8),
            )
        else:
            ax2.annotate(
                f'≤{threshold} min: {pct:.1f}%',
                xy=(threshold, pct),
                xytext=(threshold + label_offset[0], pct + label_offset[1]),
                fontsize=8,
                color=CDF_COLOR,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'),
                arrowprops=dict(arrowstyle='->', color=CDF_COLOR, alpha=0.6, linewidth=0.8),
            )
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Percentage of Buses per Bin (%)', color=MAIN_BLUE)
    ax2.set_ylabel('Cumulative Percentage (%)', color=CDF_COLOR)
    ax.set_title('(A) Delay Distribution with CDF')
    ax.set_xlim(delay_range)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax.tick_params(axis='y', labelcolor=MAIN_BLUE)
    ax2.tick_params(axis='y', labelcolor=CDF_COLOR)
    ax.grid(alpha=0.3)
    
    # Stats annotation - bottom right corner
    late_pct = (delays > 0).mean() * 100
    late_2min = (delays > 2).mean() * 100
    stats_text = f'n = {len(delays):,}\nMean = {delays.mean():.2f} min\nMedian = {delays.median():.2f} min\nLate (>2 min) = {late_2min:.1f}%'
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Panel B: Top 10 Most Delayed Lines by Average Delay
    # =========================================================================
    ax = axes[0, 1]
    
    # Calculate mean delay per line (filter to lines with sufficient data)
    line_stats = df.groupby('line_name')['delay_minutes'].agg(['mean', 'count']).reset_index()
    line_stats = line_stats[line_stats['count'] >= 100]  # min 100 observations
    line_stats = line_stats.sort_values('mean', ascending=False).head(10)
    
    bars = ax.barh(line_stats['line_name'].astype(str), line_stats['mean'], 
                   color=MAIN_BLUE, edgecolor='white')
    
    # Add delay labels
    for bar, (_, row) in zip(bars, line_stats.iterrows()):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{row["mean"]:.1f} min (n={row["count"]:,})', va='center', fontsize=8, color='black')
    
    ax.set_xlabel('Mean Delay (minutes)')
    ax.set_ylabel('Line')
    ax.set_title('(B) Top 10 Most Delayed Lines')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    # Extend x-axis to fit labels
    ax.set_xlim(0, line_stats['mean'].max() * 1.5)
    
    # =========================================================================
    # Panel C: Top 10 Busiest Stops by Departure Count
    # =========================================================================
    ax = axes[1, 0]
    
    stop_counts = df.groupby('stop_name').size().reset_index(name='count')
    stop_counts = stop_counts.sort_values('count', ascending=False).head(10)
    
    # Shorten stop names for display
    stop_counts['display_name'] = stop_counts['stop_name'].str.replace('Tübingen ', '', regex=False)
    stop_counts['display_name'] = stop_counts['display_name'].str[:25]
    
    bars = ax.barh(stop_counts['display_name'], stop_counts['count'], 
                   color=MAIN_BLUE, edgecolor='white')
    
    # Add count labels
    for bar, count in zip(bars, stop_counts['count']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', fontsize=8, color='black')
    
    ax.set_xlabel('Number of Departures')
    ax.set_ylabel('')
    ax.set_title('(C) Top 10 Busiest Stops')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    # Extend x-axis to fit labels
    ax.set_xlim(0, stop_counts['count'].max() * 1.25)
    
    # =========================================================================
    # Panel D: Mean Delay by Weather Condition with Sample Sizes
    # =========================================================================
    ax = axes[1, 1]
    
    # Check for weather column - 'condition' contains dry/rain/snow/fog/hail/sleet
    weather_col = None
    for col in ['condition', 'weather_condition', 'weather', 'precipitation']:
        if col in df.columns:
            weather_col = col
            break
    
    if weather_col is not None:
        weather_stats = df.groupby(weather_col)['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
        weather_stats = weather_stats[weather_stats['count'] >= 50]  # filter small samples
        weather_stats['ci95'] = 1.96 * weather_stats['std'] / np.sqrt(weather_stats['count'])
        weather_stats = weather_stats.sort_values('mean', ascending=False)
        
        x = range(len(weather_stats))
        bars = ax.bar(x, weather_stats['mean'], color=MAIN_BLUE, alpha=0.8, edgecolor='white')
        
        # Error bars
        yerr_lower = weather_stats['ci95']
        yerr_upper = weather_stats['ci95']
        ax.errorbar(x, weather_stats['mean'], yerr=[yerr_lower, yerr_upper],
                    fmt='none', color='black', capsize=4, capthick=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(weather_stats[weather_col], rotation=45, ha='right')
        
        # Add sample size annotations
        for i, row in weather_stats.reset_index().iterrows():
            ax.text(i, row['mean'] + row['ci95'] + 0.3, f'n={row["count"]:,}', 
                    ha='center', fontsize=8, color='black')
        
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Mean Delay (minutes)')
        ax.set_title('(D) Mean Delay by Weather Condition')
        ax.grid(axis='y', alpha=0.3)
        # Extend y-axis to fit labels
        max_y = (weather_stats['mean'] + weather_stats['ci95']).max()
        ax.set_ylim(0, max_y * 1.25)
    else:
        # Fallback: show message if no weather data
        ax.text(0.5, 0.5, 'Weather data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('(D) Mean Delay by Weather Condition')
    
    # =========================================================================
    # Save figure
    # =========================================================================
    plt.tight_layout()
    
    # Save to plots/ directory
    out_png = PLOT_DIR / "fig1_eda_4panel.png"
    out_pdf = PLOT_DIR / "fig1_eda_4panel.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    
    # Also save to paper/images/ for LaTeX
    paper_pdf = PAPER_DIR / "fig1_eda_4panel.pdf"
    plt.savefig(paper_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved: {paper_pdf}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total records: {len(delays):,}")
    print(f"Mean delay: {delays.mean():.2f} min")
    print(f"Median delay: {delays.median():.2f} min")
    print(f"Late (>0 min): {(delays > 0).mean():.1%}")
    print(f"Late (>2 min): {(delays > 2).mean():.1%}")


if __name__ == "__main__":
    main()
