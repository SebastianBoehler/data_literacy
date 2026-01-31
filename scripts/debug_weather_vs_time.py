"""
Debug script: Weather vs Time-of-Day Effect Analysis

Analyzes the relative importance of weather conditions vs time of day
on bus delays, with special focus on snow effects in pre/post schedule change.

Key questions:
1. How does weather effect compare to time-of-day effect?
2. Does snow have the largest weather effect?
3. How does snow affect pre vs post schedule change datasets?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Key dates
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")


def load_data():
    """Load trip data."""
    trip_file = OUTPUT_DIR / "all_trip_data.parquet"
    df = pd.read_parquet(trip_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['delay_minutes'].notna()].copy()
    
    # Create pre/post flag
    df['period'] = np.where(df['timestamp'] < SCHEDULE_CHANGE_DATE, 'pre', 'post')
    
    # Ensure hour column
    if 'hour' not in df.columns or df['hour'].isna().all():
        df['hour'] = df['timestamp'].dt.hour
    
    return df


def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def compute_eta_squared(df, group_col, value_col):
    """Compute eta-squared (variance explained) for a categorical grouping."""
    groups = df.groupby(group_col)[value_col]
    grand_mean = df[value_col].mean()
    
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    ss_total = ((df[value_col] - grand_mean)**2).sum()
    
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def bootstrap_ci(x, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Return (low, high) bootstrap CI."""
    x = pd.Series(x).dropna().values
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats_arr = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats_arr[i] = stat_fn(sample)
    alpha = 1 - ci
    return (np.quantile(stats_arr, alpha / 2), np.quantile(stats_arr, 1 - alpha / 2))


def main():
    print("=" * 70)
    print("WEATHER vs TIME-OF-DAY EFFECT ANALYSIS")
    print("=" * 70)
    
    df = load_data()
    print(f"\nLoaded {len(df):,} observations")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Pre-change: {(df['period'] == 'pre').sum():,} | Post-change: {(df['period'] == 'post').sum():,}")
    
    # =========================================================================
    # 1. WEATHER CONDITION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. WEATHER CONDITION EFFECTS")
    print("=" * 70)
    
    # Weather condition distribution
    print("\nWeather condition distribution:")
    condition_counts = df['condition'].value_counts(dropna=False)
    for cond, count in condition_counts.items():
        pct = count / len(df) * 100
        mean_delay = df[df['condition'] == cond]['delay_minutes'].mean()
        print(f"  {str(cond):12s}: {count:6,} ({pct:5.1f}%) | mean delay: {mean_delay:.2f} min")
    
    # Weather effect by condition
    print("\nDelay by weather condition (with 95% CI):")
    weather_stats = []
    baseline_condition = 'dry'
    baseline_delay = df[df['condition'] == baseline_condition]['delay_minutes']
    
    for cond in df['condition'].dropna().unique():
        subset = df[df['condition'] == cond]['delay_minutes']
        mean_d = subset.mean()
        ci_low, ci_high = bootstrap_ci(subset)
        n = len(subset)
        
        # Effect size vs dry
        if cond != baseline_condition and len(baseline_delay) > 1 and len(subset) > 1:
            effect_d = compute_effect_size(subset, baseline_delay)
        else:
            effect_d = 0.0
        
        weather_stats.append({
            'condition': cond,
            'n': n,
            'mean_delay': mean_d,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'effect_vs_dry': effect_d
        })
        print(f"  {cond:12s}: {mean_d:6.2f} min (CI: [{ci_low:.2f}, {ci_high:.2f}]), n={n:,}, Cohen's d vs dry: {effect_d:+.3f}")
    
    weather_df = pd.DataFrame(weather_stats)
    
    # Eta-squared for weather
    df_weather_valid = df[df['condition'].notna()]
    eta2_weather = compute_eta_squared(df_weather_valid, 'condition', 'delay_minutes')
    print(f"\n  η² (variance explained by weather condition): {eta2_weather:.4f} ({eta2_weather*100:.2f}%)")
    
    # =========================================================================
    # 2. TIME-OF-DAY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. TIME-OF-DAY EFFECTS")
    print("=" * 70)
    
    # Hour effect
    print("\nDelay by hour of day:")
    hour_stats = df.groupby('hour')['delay_minutes'].agg(['mean', 'std', 'count']).reset_index()
    
    # Find peak and off-peak hours
    peak_hour = hour_stats.loc[hour_stats['mean'].idxmax(), 'hour']
    offpeak_hour = hour_stats.loc[hour_stats['mean'].idxmin(), 'hour']
    
    peak_delay = df[df['hour'] == peak_hour]['delay_minutes']
    offpeak_delay = df[df['hour'] == offpeak_hour]['delay_minutes']
    
    print(f"  Peak hour ({int(peak_hour):02d}:00): mean={peak_delay.mean():.2f} min")
    print(f"  Off-peak hour ({int(offpeak_hour):02d}:00): mean={offpeak_delay.mean():.2f} min")
    print(f"  Difference: {peak_delay.mean() - offpeak_delay.mean():.2f} min")
    print(f"  Cohen's d (peak vs off-peak): {compute_effect_size(peak_delay, offpeak_delay):.3f}")
    
    # Eta-squared for hour
    df_hour_valid = df[df['hour'].notna()]
    eta2_hour = compute_eta_squared(df_hour_valid, 'hour', 'delay_minutes')
    print(f"\n  η² (variance explained by hour): {eta2_hour:.4f} ({eta2_hour*100:.2f}%)")
    
    # =========================================================================
    # 3. COMPARISON: WEATHER vs TIME-OF-DAY
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. COMPARISON: WEATHER vs TIME-OF-DAY")
    print("=" * 70)
    
    print(f"\n  Variance explained (η²):")
    print(f"    - Weather condition: {eta2_weather:.4f} ({eta2_weather*100:.2f}%)")
    print(f"    - Hour of day:       {eta2_hour:.4f} ({eta2_hour*100:.2f}%)")
    print(f"    - Ratio (hour/weather): {eta2_hour/eta2_weather:.1f}x")
    
    # Effect size comparison
    snow_effect = weather_df[weather_df['condition'] == 'snow']['effect_vs_dry'].values
    snow_effect = snow_effect[0] if len(snow_effect) > 0 else np.nan
    hour_effect = compute_effect_size(peak_delay, offpeak_delay)
    
    print(f"\n  Effect sizes (Cohen's d):")
    print(f"    - Snow vs Dry:           {snow_effect:+.3f}")
    print(f"    - Peak vs Off-peak hour: {hour_effect:+.3f}")
    
    if not np.isnan(snow_effect) and not np.isnan(hour_effect):
        print(f"    - Ratio (hour/snow):     {abs(hour_effect/snow_effect):.1f}x")
    
    # =========================================================================
    # 4. SNOW ANALYSIS: PRE vs POST SCHEDULE CHANGE
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. SNOW ANALYSIS: PRE vs POST SCHEDULE CHANGE")
    print("=" * 70)
    
    # Snow distribution by period
    for period in ['pre', 'post']:
        period_df = df[df['period'] == period]
        snow_count = (period_df['condition'] == 'snow').sum()
        total = len(period_df)
        snow_pct = snow_count / total * 100 if total > 0 else 0
        
        if snow_count > 0:
            snow_delay = period_df[period_df['condition'] == 'snow']['delay_minutes'].mean()
            non_snow_delay = period_df[period_df['condition'] != 'snow']['delay_minutes'].mean()
            print(f"\n  {period.upper()} schedule change:")
            print(f"    Snow observations: {snow_count:,} ({snow_pct:.1f}%)")
            print(f"    Mean delay (snow): {snow_delay:.2f} min")
            print(f"    Mean delay (non-snow): {non_snow_delay:.2f} min")
            print(f"    Snow effect: {snow_delay - non_snow_delay:+.2f} min")
        else:
            print(f"\n  {period.upper()} schedule change:")
            print(f"    Snow observations: 0 (0.0%)")
    
    # Compare overall delays pre vs post, with and without snow
    print("\n  Delay comparison (controlling for snow):")
    
    for snow_filter, label in [(True, 'Snow days only'), (False, 'Non-snow days only')]:
        pre_subset = df[(df['period'] == 'pre') & ((df['condition'] == 'snow') == snow_filter)]['delay_minutes']
        post_subset = df[(df['period'] == 'post') & ((df['condition'] == 'snow') == snow_filter)]['delay_minutes']
        
        if len(pre_subset) > 0 and len(post_subset) > 0:
            print(f"\n    {label}:")
            print(f"      Pre:  {pre_subset.mean():.2f} min (n={len(pre_subset):,})")
            print(f"      Post: {post_subset.mean():.2f} min (n={len(post_subset):,})")
            print(f"      Diff: {post_subset.mean() - pre_subset.mean():+.2f} min")
        elif len(pre_subset) == 0:
            print(f"\n    {label}: No pre-change data")
        else:
            print(f"\n    {label}: No post-change data")
    
    # =========================================================================
    # 5. WEATHER EFFECT WITHIN EACH PERIOD
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. WEATHER EFFECT WITHIN EACH PERIOD")
    print("=" * 70)
    
    for period in ['pre', 'post']:
        period_df = df[df['period'] == period]
        period_weather_valid = period_df[period_df['condition'].notna()]
        
        if len(period_weather_valid) > 0:
            eta2 = compute_eta_squared(period_weather_valid, 'condition', 'delay_minutes')
            print(f"\n  {period.upper()} schedule change:")
            print(f"    η² (weather): {eta2:.4f} ({eta2*100:.2f}%)")
            
            # Show condition breakdown
            for cond in period_df['condition'].dropna().unique():
                subset = period_df[period_df['condition'] == cond]['delay_minutes']
                if len(subset) > 0:
                    print(f"      {cond:12s}: {subset.mean():6.2f} min (n={len(subset):,})")
    
    # =========================================================================
    # 6. SUMMARY TABLE FOR PAPER
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. SUMMARY TABLE (FOR PAPER)")
    print("=" * 70)
    
    print("\n  | Factor           | η² (Var. Explained) | Effect Size (d) |")
    print("  |------------------|---------------------|-----------------|")
    print(f"  | Hour of Day      | {eta2_hour:.4f} ({eta2_hour*100:5.2f}%)     | {hour_effect:+.3f}           |")
    print(f"  | Weather Cond.    | {eta2_weather:.4f} ({eta2_weather*100:5.2f}%)     | {snow_effect:+.3f} (snow)    |")
    
    ratio = eta2_hour / eta2_weather if eta2_weather > 0 else float('inf')
    print(f"\n  -> Time-of-day explains {ratio:.1f}x more variance than weather")
    
    # =========================================================================
    # 7. KEY INSIGHT: POST-CHANGE BETTER DESPITE SNOW
    # =========================================================================
    print("\n" + "=" * 70)
    print("7. KEY INSIGHT: POST-CHANGE PERFORMANCE DESPITE SNOW")
    print("=" * 70)
    
    pre_mean = df[df['period'] == 'pre']['delay_minutes'].mean()
    post_mean = df[df['period'] == 'post']['delay_minutes'].mean()
    
    pre_snow_pct = (df[df['period'] == 'pre']['condition'] == 'snow').mean() * 100
    post_snow_pct = (df[df['period'] == 'post']['condition'] == 'snow').mean() * 100
    
    print(f"\n  Pre-change:  mean delay = {pre_mean:.2f} min, snow = {pre_snow_pct:.1f}%")
    print(f"  Post-change: mean delay = {post_mean:.2f} min, snow = {post_snow_pct:.1f}%")
    print(f"  Improvement: {pre_mean - post_mean:.2f} min ({(pre_mean - post_mean)/pre_mean*100:.1f}%)")
    
    if post_snow_pct > pre_snow_pct:
        print(f"\n  ⚠️  Post-change period has MORE snow ({post_snow_pct:.1f}% vs {pre_snow_pct:.1f}%)")
        print(f"      Yet delays are LOWER - schedule change effect is robust!")
        print(f"      Without snow, post-change performance would likely be even better.")
    else:
        print(f"\n  Pre-change period has more snow ({pre_snow_pct:.1f}% vs {post_snow_pct:.1f}%)")
    
    # Estimate snow-adjusted improvement
    snow_effect_val = 0
    if 'snow' in weather_df['condition'].values:
        snow_row = weather_df[weather_df['condition'] == 'snow'].iloc[0]
        dry_row = weather_df[weather_df['condition'] == 'dry']
        if len(dry_row) > 0:
            snow_effect_val = snow_row['mean_delay'] - dry_row.iloc[0]['mean_delay']
    
    if snow_effect_val != 0 and post_snow_pct > pre_snow_pct:
        snow_penalty = (post_snow_pct - pre_snow_pct) / 100 * snow_effect_val
        adjusted_improvement = (pre_mean - post_mean) + snow_penalty
        print(f"\n  Snow-adjusted improvement estimate:")
        print(f"    Raw improvement: {pre_mean - post_mean:.2f} min")
        print(f"    Snow penalty (extra snow in post): {snow_penalty:.2f} min")
        print(f"    Adjusted improvement: {adjusted_improvement:.2f} min")


if __name__ == "__main__":
    main()
