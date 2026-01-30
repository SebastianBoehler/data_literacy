#!/usr/bin/env python3
"""
Figure 2: Schedule Change vs Holiday Effect Analysis

Analyzes whether the improvement in bus delays after December 14, 2025
is due to the schedule change or the holiday period (Dec 28 - Jan 5).

Outputs:
- plots/fig2_schedule_change.png
- plots/fig2_schedule_change.pdf
- outputs/fig2_schedule_change.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
    print("tueplots not available, using default styling")

SCRIPT_DIR = Path(__file__).parent.parent  # scripts/ -> code/
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOT_DIR = SCRIPT_DIR / "plots"

# Key dates
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")
HOLIDAY_START = pd.Timestamp("2025-12-28")
HOLIDAY_END = pd.Timestamp("2026-01-05")

def load_data():
    """Load trip data with actual observed delays from parquet file.
    
    The exported parquet contains only phase=='previous' stops (actual observed delays),
    deduplicated by journey/day/stop. The 'delay_minutes' column is already set.
    """
    trip_file = OUTPUT_DIR / "all_trip_data.parquet"
    if trip_file.exists():
        print(f"Loading trip data from {trip_file}")
        df = pd.read_parquet(trip_file)
        # delay_minutes should already be set in the exported data
        if 'delay_minutes' not in df.columns:
            # Fallback for older exports
            if 'departure_delay_minutes' in df.columns:
                df['delay_minutes'] = df['departure_delay_minutes']
            elif 'arrival_delay_minutes' in df.columns:
                df['delay_minutes'] = df['arrival_delay_minutes']
        print(f"  Data contains ACTUAL OBSERVED DELAYS (phase='previous' only)")
        return df, 'trip'
    
    # Fallback to departure data
    departure_files = list(OUTPUT_DIR.glob("departure_data_*.parquet"))
    if departure_files:
        print(f"Loading {len(departure_files)} departure files")
        dfs = [pd.read_parquet(f) for f in departure_files]
        df = pd.concat(dfs, ignore_index=True)
        return df, 'departure'
    
    raise FileNotFoundError("No data files found in outputs/")

def bootstrap_ci(x, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Return (low, high) bootstrap CI for a 1D array-like x."""
    x = pd.Series(x).dropna().values
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats[i] = stat_fn(sample)
    alpha = 1 - ci
    return (np.quantile(stats, alpha / 2), np.quantile(stats, 1 - alpha / 2))

def categorize_period(ts):
    """Categorize a timestamp into analysis periods."""
    if pd.isna(ts):
        return "Unknown"
    if ts < SCHEDULE_CHANGE_DATE:
        return "1_Before_Schedule_Change"
    elif ts >= SCHEDULE_CHANGE_DATE and ts < HOLIDAY_START:
        return "2_After_Change_Before_Holiday"
    elif ts >= HOLIDAY_START and ts <= HOLIDAY_END:
        return "3_Holiday_Period"
    else:
        return "4_After_Holiday"

def main():
    print("=" * 60)
    print("HOLIDAY vs SCHEDULE CHANGE ANALYSIS")
    print("=" * 60)
    
    # Load data
    df, data_type = load_data()
    print(f"\nLoaded {len(df):,} rows ({data_type} data)")
    
    # Ensure timestamp column
    ts_col = None
    for col in ['timestamp', 'scheduled_time', 'planned_time']:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        print("ERROR: No timestamp column found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"Using timestamp column: {ts_col}")
    
    # Convert to datetime
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df['date'] = df[ts_col].dt.date
    
    # Filter to valid delay data
    if 'delay_minutes' not in df.columns:
        print("ERROR: No delay_minutes column!")
        return
    
    df = df[df['delay_minutes'].notna()].copy()
    print(f"After filtering NaN delays: {len(df):,} rows")
    
    # Date range
    min_date = df[ts_col].min()
    max_date = df[ts_col].max()
    print(f"\nDate range: {min_date.date()} to {max_date.date()}")
    
    # Categorize periods
    df['period'] = df[ts_col].apply(categorize_period)
    
    # Summary by period
    print("\n" + "=" * 60)
    print("DELAY STATISTICS BY PERIOD")
    print("=" * 60)
    
    results = []
    for period in sorted(df['period'].unique()):
        subset = df[df['period'] == period]['delay_minutes']
        n = len(subset)
        mean_delay = subset.mean()
        median_delay = subset.median()
        ci_low, ci_high = bootstrap_ci(subset)
        p90 = subset.quantile(0.90)
        late_pct = (subset > 0).mean() * 100
        
        results.append({
            'period': period,
            'n': n,
            'mean': mean_delay,
            'median': median_delay,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'p90': p90,
            'late_pct': late_pct
        })
        
        print(f"\n{period}:")
        print(f"  N = {n:,}")
        print(f"  Mean delay: {mean_delay:.3f} min (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
        print(f"  Median delay: {median_delay:.3f} min")
        print(f"  P90 delay: {p90:.2f} min")
        print(f"  % Late (>0 min): {late_pct:.1f}%")
    
    results_df = pd.DataFrame(results)
    
    # Daily analysis
    print("\n" + "=" * 60)
    print("DAILY MEAN DELAY AROUND KEY DATES")
    print("=" * 60)
    
    daily = df.groupby('date')['delay_minutes'].agg(['mean', 'count']).reset_index()
    daily.columns = ['date', 'mean_delay', 'n_obs']
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Filter to relevant period (2 weeks before/after schedule change)
    analysis_start = SCHEDULE_CHANGE_DATE - pd.Timedelta(days=14)
    analysis_end = HOLIDAY_END + pd.Timedelta(days=14)
    daily_filtered = daily[(daily['date'] >= analysis_start) & (daily['date'] <= analysis_end)]
    
    print(f"\nDaily data from {analysis_start.date()} to {analysis_end.date()}:")
    for _, row in daily_filtered.iterrows():
        marker = ""
        if row['date'].date() == SCHEDULE_CHANGE_DATE.date():
            marker = " <-- SCHEDULE CHANGE"
        elif row['date'].date() == HOLIDAY_START.date():
            marker = " <-- HOLIDAY START"
        elif row['date'].date() == HOLIDAY_END.date():
            marker = " <-- HOLIDAY END"
        print(f"  {row['date'].date()}: mean={row['mean_delay']:.3f} min (n={row['n_obs']:,}){marker}")
    
    # Weekday analysis (exclude holidays)
    print("\n" + "=" * 60)
    print("WEEKDAY vs WEEKEND (EXCLUDING HOLIDAYS)")
    print("=" * 60)
    
    df['weekday'] = df[ts_col].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6])
    df['is_holiday'] = df['period'] == "3_Holiday_Period"
    
    # Non-holiday weekdays vs weekends
    non_holiday = df[~df['is_holiday']]
    
    for is_wknd, label in [(False, "Weekday"), (True, "Weekend")]:
        subset = non_holiday[non_holiday['is_weekend'] == is_wknd]['delay_minutes']
        mean_d = subset.mean()
        ci = bootstrap_ci(subset)
        print(f"{label} (non-holiday): mean={mean_d:.3f} min (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]), n={len(subset):,}")
    
    # Create visualization - 2 panel layout (time series + mean delay)
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    # Use a professional color palette (muted)
    colors_periods = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f']  # muted blue, orange, green, red
    
    fig, axes_arr = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # Plot 1: Daily mean delay time series with shaded holiday region
    ax = axes_arr[0]
    ax.plot(daily_filtered['date'], daily_filtered['mean_delay'], 'o-', 
            markersize=3, alpha=0.8, color='#4878d0', linewidth=1)
    
    # Shaded region for holiday period
    ax.axvspan(HOLIDAY_START, HOLIDAY_END, alpha=0.2, color='#6acc64', label='Holiday Period')
    
    # Vertical line for schedule change
    ax.axvline(SCHEDULE_CHANGE_DATE, color='#d65f5f', linestyle='--', linewidth=1.5, label='Schedule Change (Dec 14)')
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Delay (min)')
    ax.set_title('(A) Daily Mean Delay')
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Hourly delay pattern comparing pre/post schedule change
    ax = axes_arr[1]
    
    # Split data by schedule change
    df_pre = df[df['period'] == '1_Before_Schedule_Change'].copy()
    df_post = df[df['period'].isin(['2_After_Change_Before_Holiday', '4_After_Holiday'])].copy()
    
    # Compute hourly stats for each period
    df_pre['hour'] = df_pre[ts_col].dt.hour
    df_post['hour'] = df_post[ts_col].dt.hour
    df['hour'] = df[ts_col].dt.hour
    
    pre_hourly = df_pre.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    pre_hourly['ci95'] = 1.96 * pre_hourly['std'] / np.sqrt(pre_hourly['count'])
    
    post_hourly = df_post.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    post_hourly['ci95'] = 1.96 * post_hourly['std'] / np.sqrt(post_hourly['count'])
    
    # Total hourly (all data)
    total_hourly = df.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    total_hourly['ci95'] = 1.96 * total_hourly['std'] / np.sqrt(total_hourly['count'])
    
    # Total (circles, blue) - plot first so it's in background
    ax.fill_between(total_hourly['hour'], 
                    total_hourly['mean'] - total_hourly['ci95'],
                    total_hourly['mean'] + total_hourly['ci95'],
                    alpha=0.1, color='#4878d0')
    ax.plot(total_hourly['hour'], total_hourly['mean'], 'o-', color='#4878d0', 
            linewidth=1.5, markersize=4, alpha=0.6, label=f'Total (μ={df["delay_minutes"].mean():.2f})')
    
    # Pre schedule change (squares, red)
    ax.fill_between(pre_hourly['hour'], 
                    pre_hourly['mean'] - pre_hourly['ci95'],
                    pre_hourly['mean'] + pre_hourly['ci95'],
                    alpha=0.15, color='#d65f5f')
    ax.plot(pre_hourly['hour'], pre_hourly['mean'], 's--', color='#d65f5f', 
            linewidth=1.5, markersize=5, alpha=0.8, label=f'Pre-Change (μ={df_pre["delay_minutes"].mean():.2f})')
    
    # Post schedule change (triangles, green)
    ax.fill_between(post_hourly['hour'], 
                    post_hourly['mean'] - post_hourly['ci95'],
                    post_hourly['mean'] + post_hourly['ci95'],
                    alpha=0.15, color='#6acc64')
    ax.plot(post_hourly['hour'], post_hourly['mean'], '^-', color='#6acc64', 
            linewidth=1.5, markersize=5, alpha=0.8, label=f'Post-Change (μ={df_post["delay_minutes"].mean():.2f})')
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Delay (min)')
    ax.set_title('(B) Hourly Delay Pattern: Pre vs Post Schedule Change')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    
    # Save with consistent naming (fig2_schedule_change)
    out_png = PLOT_DIR / "fig2_schedule_change.png"
    out_pdf = PLOT_DIR / "fig2_schedule_change.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    
    # Save CSV with same name
    csv_out = OUTPUT_DIR / "fig2_schedule_change.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    
    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    before = results_df[results_df['period'] == '1_Before_Schedule_Change']['mean'].values[0]
    after_pre_holiday = results_df[results_df['period'] == '2_After_Change_Before_Holiday']['mean'].values
    holiday = results_df[results_df['period'] == '3_Holiday_Period']['mean'].values
    after_holiday = results_df[results_df['period'] == '4_After_Holiday']['mean'].values
    
    print(f"\n1. Before schedule change: {before:.3f} min mean delay")
    
    if len(after_pre_holiday) > 0:
        print(f"2. After change, before holiday: {after_pre_holiday[0]:.3f} min mean delay")
        print(f"   -> Change from before: {after_pre_holiday[0] - before:.3f} min ({(after_pre_holiday[0] - before) / before * 100:.1f}%)")
    
    if len(holiday) > 0:
        print(f"3. During holiday period: {holiday[0]:.3f} min mean delay")
    
    if len(after_holiday) > 0:
        print(f"4. After holiday: {after_holiday[0]:.3f} min mean delay")
        print(f"   -> This is the 'true' post-change effect (excluding holiday confound)")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION NOTES")
    print("=" * 60)
    print("""
If delay is lower during holidays AND after holidays compared to before:
  -> Schedule change likely had a real effect

If delay is lower ONLY during holidays:
  -> Improvement is likely due to reduced traffic/ridership, not schedule change

If delay after holidays returns to pre-change levels:
  -> Schedule change had no lasting effect; holiday effect was temporary
""")

if __name__ == "__main__":
    main()
