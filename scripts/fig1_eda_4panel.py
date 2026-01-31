"""
Figure 1: Exploratory Analysis of Tübingen Bus Delays (4-Panel)

Generates a 2x2 figure with:
(A) Distribution of departure delays (right-skewed pattern)
(B) Mean delay by hour of day with 95% CI
(C) Top 10 busiest stops by departure count
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
    # Panel A: Distribution of Departure Delays
    # =========================================================================
    ax = axes[0, 0]
    
    ax.hist(delays, bins=50, range=(-1, 20), color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(delays.mean(), color='orange', linestyle='-', linewidth=1.5, 
               label=f'Mean: {delays.mean():.2f} min')
    ax.axvline(delays.median(), color='green', linestyle='-', linewidth=1.5, 
               label=f'Median: {delays.median():.2f} min')
    
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('(A) Distribution of Departure Delays')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('log')
    ax.set_xlim(-1, 20)
    ax.grid(alpha=0.3)
    
    # Stats annotation
    late_pct = (delays > 0).mean() * 100
    stats_text = f'n = {len(delays):,}\nLate (>0): {late_pct:.1f}%'
    ax.text(0.97, 0.75, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Panel B: Mean Delay by Hour of Day with 95% CI
    # =========================================================================
    ax = axes[0, 1]
    
    df_hourly = df.copy()
    df_hourly['hour'] = pd.to_datetime(df_hourly['timestamp']).dt.hour
    
    hourly_stats = df_hourly.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    hourly_stats['ci95'] = 1.96 * hourly_stats['std'] / np.sqrt(hourly_stats['count'])
    
    ax.fill_between(hourly_stats['hour'], 
                    hourly_stats['mean'] - hourly_stats['ci95'],
                    hourly_stats['mean'] + hourly_stats['ci95'],
                    alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(hourly_stats['hour'], hourly_stats['mean'], 'o-', color='steelblue', 
            linewidth=1.5, markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Delay (minutes)')
    ax.set_title('(B) Mean Delay by Hour of Day')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
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
                   color='steelblue', edgecolor='white')
    
    # Add count labels
    for bar, count in zip(bars, stop_counts['count']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', fontsize=7)
    
    ax.set_xlabel('Number of Departures')
    ax.set_ylabel('')
    ax.set_title('(C) Top 10 Busiest Stops')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
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
        bars = ax.bar(x, weather_stats['mean'], color='steelblue', alpha=0.7, edgecolor='black')
        
        # Error bars
        yerr_lower = weather_stats['ci95']
        yerr_upper = weather_stats['ci95']
        ax.errorbar(x, weather_stats['mean'], yerr=[yerr_lower, yerr_upper],
                    fmt='none', color='black', capsize=4, capthick=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(weather_stats[weather_col], rotation=45, ha='right', fontsize=8)
        
        # Add sample size annotations
        for i, row in weather_stats.reset_index().iterrows():
            ax.text(i, row['mean'] + row['ci95'] + 0.3, f'n={row["count"]:,}', 
                    ha='center', fontsize=6, color='gray')
        
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Mean Delay (minutes)')
        ax.set_title('(D) Mean Delay by Weather Condition')
        ax.grid(axis='y', alpha=0.3)
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
