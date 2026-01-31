"""
Pre/Post Schedule Change ECDF Comparison

Compares delay distributions before and after the Dec 14, 2025 schedule change
using ECDF overlays with DKW confidence bands and key metrics with bootstrap CIs.

Output: outputs/schedule_change_ecdf_comparison.png
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

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "schedule_change_ecdf_comparison.png"

# Constants
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")
LATE_THRESHOLD = 2  # minutes
DKW_ALPHA = 0.05  # 95% confidence
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def bootstrap_ci(data, func, n_boot=N_BOOTSTRAP):
    """Calculate bootstrap confidence interval."""
    stats = [func(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def main():
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Split by period
    pre = df[df['timestamp'] < SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna()
    post = df[df['timestamp'] >= SCHEDULE_CHANGE_DATE]['delay_minutes'].dropna()

    print(f"Pre-schedule: {len(pre):,} records")
    print(f"Post-schedule: {len(post):,} records")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ECDF comparison
    ax1 = axes[0]

    # Pre ECDF
    pre_sorted = np.sort(pre)
    pre_ecdf = np.arange(1, len(pre_sorted) + 1) / len(pre_sorted)
    step_pre = max(1, len(pre_sorted) // 3000)

    # Post ECDF
    post_sorted = np.sort(post)
    post_ecdf = np.arange(1, len(post_sorted) + 1) / len(post_sorted)
    step_post = max(1, len(post_sorted) // 3000)

    ax1.plot(pre_sorted[::step_pre], pre_ecdf[::step_pre],
             color='steelblue', linewidth=2, label=f'Before (n={len(pre):,})')
    ax1.plot(post_sorted[::step_post], post_ecdf[::step_post],
             color='darkorange', linewidth=2, label=f'After (n={len(post):,})')

    # DKW bands
    eps_pre = np.sqrt(np.log(2 / DKW_ALPHA) / (2 * len(pre)))
    eps_post = np.sqrt(np.log(2 / DKW_ALPHA) / (2 * len(post)))

    ax1.fill_between(
        pre_sorted[::step_pre],
        np.clip(pre_ecdf[::step_pre] - eps_pre, 0, 1),
        np.clip(pre_ecdf[::step_pre] + eps_pre, 0, 1),
        alpha=0.15, color='steelblue'
    )
    ax1.fill_between(
        post_sorted[::step_post],
        np.clip(post_ecdf[::step_post] - eps_post, 0, 1),
        np.clip(post_ecdf[::step_post] + eps_post, 0, 1),
        alpha=0.15, color='darkorange'
    )

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(LATE_THRESHOLD, color='red', linestyle=':', alpha=0.7,
                label=f'Late threshold ({LATE_THRESHOLD} min)')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('Delay Distribution: Before vs After Schedule Change (Dec 14, 2025)')
    ax1.set_xlim(-5, 20)
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)

    # Right: Key metrics comparison with CI
    ax2 = axes[1]

    metrics = ['P(on time)', 'P(late > 2min)', 'Median delay']
    pre_vals = [(pre <= 0).mean(), (pre > LATE_THRESHOLD).mean(), np.median(pre)]
    post_vals = [(post <= 0).mean(), (post > LATE_THRESHOLD).mean(), np.median(post)]

    # Bootstrap CIs
    pre_cis = [
        bootstrap_ci(pre.values, lambda x: (x <= 0).mean()),
        bootstrap_ci(pre.values, lambda x: (x > LATE_THRESHOLD).mean()),
        bootstrap_ci(pre.values, np.median),
    ]
    post_cis = [
        bootstrap_ci(post.values, lambda x: (x <= 0).mean()),
        bootstrap_ci(post.values, lambda x: (x > LATE_THRESHOLD).mean()),
        bootstrap_ci(post.values, np.median),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    # For first two metrics, multiply by 100 for percentage
    pre_plot = [pre_vals[0] * 100, pre_vals[1] * 100, pre_vals[2]]
    post_plot = [post_vals[0] * 100, post_vals[1] * 100, post_vals[2]]

    pre_err = [
        [pre_vals[i] * 100 - pre_cis[i][0] * 100 if i < 2 else pre_vals[i] - pre_cis[i][0] for i in range(3)],
        [pre_cis[i][1] * 100 - pre_vals[i] * 100 if i < 2 else pre_cis[i][1] - pre_vals[i] for i in range(3)]
    ]
    post_err = [
        [post_vals[i] * 100 - post_cis[i][0] * 100 if i < 2 else post_vals[i] - post_cis[i][0] for i in range(3)],
        [post_cis[i][1] * 100 - post_vals[i] * 100 if i < 2 else post_cis[i][1] - post_vals[i] for i in range(3)]
    ]

    bars1 = ax2.bar(x - width / 2, pre_plot, width, label='Before', color='steelblue', alpha=0.7)
    bars2 = ax2.bar(x + width / 2, post_plot, width, label='After', color='darkorange', alpha=0.7)

    ax2.errorbar(x - width / 2, pre_plot, yerr=pre_err, fmt='none', color='black', capsize=4)
    ax2.errorbar(x + width / 2, post_plot, yerr=post_err, fmt='none', color='black', capsize=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value (% or minutes)')
    ax2.set_title('Key Metrics: Before vs After Schedule Change')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, pre_plot):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, post_plot):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")

    # Print findings
    print(f'\nKey findings:')
    print(f'  Before: {pre_vals[0]:.1%} on time, {pre_vals[1]:.1%} late, median {pre_vals[2]:.1f} min')
    print(f'  After:  {post_vals[0]:.1%} on time, {post_vals[1]:.1%} late, median {post_vals[2]:.1f} min')
    print(f'\nChange in late rate: {pre_vals[1]:.1%} → {post_vals[1]:.1%} '
          f'(Δ = {(post_vals[1] - pre_vals[1]) * 100:+.1f} percentage points)')


if __name__ == "__main__":
    main()
