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


def compute_variance_explained(df: pd.DataFrame) -> dict:
    """Compute R² for time of day and weather condition on delay variance."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LinearRegression
    
    df_clean = df.dropna(subset=['delay_minutes', 'condition', 'hour']).copy()
    df_clean['hour_int'] = df_clean['hour'].astype(int)
    
    le = LabelEncoder()
    df_clean['condition_enc'] = le.fit_transform(df_clean['condition'])
    
    y = df_clean['delay_minutes'].values
    
    # R² for hour only
    X_hour = df_clean[['hour_int']].values
    r2_hour = LinearRegression().fit(X_hour, y).score(X_hour, y)
    
    # R² for condition only
    X_cond = df_clean[['condition_enc']].values
    r2_cond = LinearRegression().fit(X_cond, y).score(X_cond, y)
    
    return {
        'r2_hour': r2_hour,
        'r2_weather': r2_cond,
        'ratio': r2_cond / r2_hour if r2_hour > 0 else float('inf'),
    }


def generate_weather_vs_time_plot(df: pd.DataFrame, period: str, period_label: str, output_dir: Path,
                                   snow_note: str = None):
    """Generate bar chart comparing variance explained by weather vs time of day.
    
    Simple visualization showing which factor explains more delay variance.
    """
    # Filter valid data
    df_valid = df.dropna(subset=['delay_minutes', 'condition', 'hour']).copy()
    
    if len(df_valid) < 100:
        print(f"  Skipped weather_vs_time.png (insufficient data)")
        return
    
    # Compute variance explained
    var_stats = compute_variance_explained(df_valid)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data for bar chart
    factors = ['Weather\nCondition', 'Time of\nDay']
    r2_values = [var_stats['r2_weather'] * 100, var_stats['r2_hour'] * 100]
    colors = ['steelblue', 'darkorange']
    
    # Create bars
    bars = ax.bar(factors, r2_values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Variance Explained (R²)', fontsize=12)
    ax.set_title(f'Weather vs Time of Day: Effect on Delays\n{period_label}', fontweight='bold', fontsize=13)
    ax.set_ylim(0, max(r2_values) * 1.4)
    ax.grid(axis='y', alpha=0.3)
    
    # Add ratio annotation
    ratio_text = f"Weather explains {var_stats['ratio']:.1f}× more variance than time of day"
    ax.text(0.5, 0.92, ratio_text, transform=ax.transAxes, fontsize=11,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    # Add snow note if provided
    if snow_note:
        ax.text(0.5, 0.02, snow_note, transform=ax.transAxes, fontsize=9,
                ha='center', va='bottom', color='gray', style='italic')
    
    # Add sample size
    ax.text(0.98, 0.98, f'n = {len(df_valid):,}', transform=ax.transAxes, fontsize=9,
            ha='right', va='top', color='gray')
    
    plt.tight_layout()
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
    
    # Check snow distribution across periods
    df_pre = filter_by_period(df, 'pre')
    df_post = filter_by_period(df, 'post')
    
    snow_pre = (df_pre['condition'] == 'snow').sum() if 'condition' in df_pre.columns else 0
    snow_post = (df_post['condition'] == 'snow').sum() if 'condition' in df_post.columns else 0
    
    print(f"\nSnow observations: pre={snow_pre}, post={snow_post}")
    
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
        
        # Add snow note for pre-change period
        snow_note = None
        if period == "pre":
            snow_note = "Note: No snow observations in pre-change period"
        elif period == "all":
            snow_note = f"Note: Snow data only from post-change\n(n={snow_post:,} obs, 0 pre-change)"
        
        generate_weather_vs_time_plot(period_df, period, config['label'], output_dir, snow_note=snow_note)
    
    print("\n" + "=" * 70)
    print("Done! Generated weather_vs_time.png for all periods")
    print("=" * 70)


if __name__ == "__main__":
    main()
