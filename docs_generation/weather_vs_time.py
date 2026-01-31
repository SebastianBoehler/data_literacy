"""
Generate Weather vs Time of Day interaction plot for GitHub Pages.

Shows how delays vary by both weather condition and time of day.

Usage:
    python docs_generation/weather_vs_time.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
SCRIPT_DIR = Path(__file__).parent.parent  # docs_generation/ -> code/
sys.path.insert(0, str(SCRIPT_DIR))
from modules.plot_config import apply_style, STYLE

apply_style()

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


def generate_weather_vs_time_heatmap(df: pd.DataFrame, period: str, period_label: str, output_dir: Path):
    """Generate heatmap showing weather condition vs time of day effect on delays.
    
    Shows late rate (P(delay > 2 min)) as a function of both hour of day and weather condition.
    """
    LATE_THRESHOLD = 2
    MIN_SAMPLES = 30
    
    # Filter valid data with weather and time info
    df_valid = df.dropna(subset=['delay_minutes', 'condition', 'hour']).copy()
    df_valid['hour'] = df_valid['hour'].astype(int)
    df_valid['is_late'] = df_valid['delay_minutes'] > LATE_THRESHOLD
    
    if len(df_valid) < 100:
        print(f"  Skipped weather_vs_time.png (insufficient data)")
        return
    
    # Get weather conditions with enough samples
    condition_counts = df_valid['condition'].value_counts()
    valid_conditions = condition_counts[condition_counts >= MIN_SAMPLES * 5].index.tolist()
    
    if len(valid_conditions) < 2:
        print(f"  Skipped weather_vs_time.png (not enough weather conditions)")
        return
    
    df_valid = df_valid[df_valid['condition'].isin(valid_conditions)]
    
    # Create pivot table: late rate by hour and condition
    pivot = df_valid.groupby(['condition', 'hour']).agg(
        late_rate=('is_late', 'mean'),
        count=('is_late', 'count')
    ).reset_index()
    
    # Filter cells with minimum samples
    pivot = pivot[pivot['count'] >= MIN_SAMPLES]
    
    # Create heatmap matrix
    conditions = sorted(df_valid['condition'].unique())
    hours = list(range(5, 24))  # Focus on 5am-11pm
    
    heatmap_data = np.full((len(conditions), len(hours)), np.nan)
    count_data = np.full((len(conditions), len(hours)), 0)
    
    for _, row in pivot.iterrows():
        if row['hour'] in hours:
            i = conditions.index(row['condition'])
            j = hours.index(row['hour'])
            heatmap_data[i, j] = row['late_rate'] * 100
            count_data[i, j] = row['count']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel A: Heatmap ---
    ax = axes[0]
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
    
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f'{h}' for h in hours], fontsize=9)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c.capitalize() for c in conditions], fontsize=10)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weather Condition')
    ax.set_title('(A) Late Rate by Weather & Time of Day', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Late Rate (%)', fontsize=10)
    
    # --- Panel B: Line plot comparison ---
    ax = axes[1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    
    for idx, condition in enumerate(conditions):
        cond_data = pivot[pivot['condition'] == condition].sort_values('hour')
        if len(cond_data) >= 3:
            ax.plot(cond_data['hour'], cond_data['late_rate'] * 100, 
                    'o-', color=colors[idx], linewidth=2, markersize=5,
                    label=f'{condition.capitalize()} (n={df_valid[df_valid["condition"]==condition].shape[0]:,})')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Late Rate (%)')
    ax.set_title('(B) Late Rate Trends by Weather Condition', fontweight='bold')
    ax.set_xlim(4.5, 23.5)
    ax.set_ylim(0, 50)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add overall stats
    overall_late = df_valid['is_late'].mean() * 100
    ax.axhline(overall_late, color='gray', linestyle='--', alpha=0.7)
    
    fig.suptitle(f'Weather vs Time of Day Effect on Delays — {period_label}', fontsize=12, fontweight='bold', y=1.02)
    
    output_path = output_dir / "weather_vs_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("Generating Weather vs Time of Day Plot")
    print("=" * 70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate plots for each period
    for period, config in PERIODS.items():
        print(f"\nPeriod: {period} ({config['label']})")
        
        # Create period-specific directory
        output_dir = PLOTS_DIR / period
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter data
        period_df = filter_by_period(df, period)
        print(f"  Records: {len(period_df):,}")
        
        if period_df.empty:
            print(f"  WARNING: No data for period '{period}'")
            continue
        
        generate_weather_vs_time_heatmap(period_df, period, config['label'], output_dir)
    
    print("\n" + "=" * 70)
    print("Done! Generated weather_vs_time.png for all periods")
    print("=" * 70)


if __name__ == "__main__":
    main()
