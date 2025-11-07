from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency for quick plots
    plt = None

from modules.utils import ensure_directory

EXPORT_DIR = Path(__file__).resolve().parent / "exports"
AGGREGATED_FILENAME = "departures_aggregated.csv"
PLOTS_SUBDIR = "plots"
RUN_PLOTS = True  # Flip to False if you only want aggregation
CUSTOM_PLOTS_DIR: Path | None = None  # Optionally store plots elsewhere

TIMESTAMP_PATTERN = re.compile(r"departures_(\d{8})_(\d{6})\.csv")


def find_departure_files(export_dir: Path) -> list[Path]:
    """Return every departure CSV in the export directory, sorted by snapshot."""

    if not export_dir.exists():
        return []
    # Filter out the aggregated file and only return timestamped snapshot files
    return sorted([
        f for f in export_dir.glob("departures_*.csv") 
        if f.name != AGGREGATED_FILENAME
    ])


def parse_snapshot_timestamp(file_path: Path) -> datetime:
    """Extract the snapshot timestamp encoded in a departures CSV name."""

    match = TIMESTAMP_PATTERN.fullmatch(file_path.name)
    if not match:
        raise ValueError(
            f"Could not parse timestamp from file '{file_path.name}'. Expected format 'departures_YYYYMMDD_HHMMSS.csv'."
        )
    date_part, time_part = match.groups()
    return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")


def _coerce_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert listed columns to datetimes if they exist."""

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert listed columns to numeric if they exist."""

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_departure_snapshots(export_dir: Path) -> pd.DataFrame:
    """Load all departure CSV files and augment them with metadata."""

    frames: list[pd.DataFrame] = []
    for csv_path in find_departure_files(export_dir):
        data = pd.read_csv(csv_path)
        if data.empty:
            continue

        snapshot_time = parse_snapshot_timestamp(csv_path)
        data["snapshot_time"] = pd.Timestamp(snapshot_time)
        data["source_file"] = csv_path.name

        _coerce_datetime_columns(data, ("planned_time", "estimated_time", "weather_timestamp"))
        _coerce_numeric_columns(
            data,
            (
                "delay_minutes",
                "temperature",
                "precipitation_mm",
                "wind_speed_ms",
                "wind_direction_deg",
                "pressure_hpa",
                "relative_humidity",
            ),
        )

        frames.append(data)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [col for col in ("snapshot_time", "stop_id", "planned_time") if col in combined.columns]
    if sort_cols:
        combined.sort_values(sort_cols, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def plot_mean_delay_by_snapshot(data: pd.DataFrame, output_dir: Path) -> None:
    """Plot how the average delay evolves per snapshot."""

    if "delay_minutes" not in data.columns:
        return

    series = (
        data.dropna(subset=["snapshot_time", "delay_minutes"])
        .groupby("snapshot_time")["delay_minutes"]
        .mean()
        .sort_index()
    )
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, marker="o")
    ax.set_title("Average delay per snapshot")
    ax.set_xlabel("Snapshot time")
    ax.set_ylabel("Average delay (minutes)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    output_path = output_dir / "mean_delay_by_snapshot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_stops_by_delay(data: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Plot stops with the highest average delay."""

    if "delay_minutes" not in data.columns or "stop_name" not in data.columns:
        return

    summary = (
        data.dropna(subset=["stop_name", "delay_minutes"])
        .groupby("stop_name")["delay_minutes"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    summary.iloc[::-1].plot(kind="barh", ax=ax, color="#C2542D")
    ax.set_title(f"Top {len(summary)} stops by average delay")
    ax.set_xlabel("Average delay (minutes)")
    ax.set_ylabel("Stop")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    output_path = output_dir / "top_stops_by_delay.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_delay_vs_weather(data: pd.DataFrame, weather_col: str, label: str, output_path: Path) -> None:
    subset = data.dropna(subset=["delay_minutes", weather_col])
    if subset.empty:
        return

    corr = subset["delay_minutes"].corr(subset[weather_col])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        subset[weather_col],
        subset["delay_minutes"],
        alpha=0.35,
        s=22,
        edgecolor="none",
        color="#2364AA",
    )

    if subset[weather_col].nunique() > 1:
        x_vals = np.linspace(subset[weather_col].min(), subset[weather_col].max(), 100)
        slope, intercept = np.polyfit(subset[weather_col], subset["delay_minutes"], deg=1)
        ax.plot(x_vals, slope * x_vals + intercept, color="#C2542D", linewidth=1.2, label="Trend")
        ax.legend(frameon=False)

    corr_text = f" (r={corr:.2f})" if pd.notna(corr) else ""
    ax.set_title(f"Delay vs {label}{corr_text}")
    ax.set_xlabel(label)
    ax.set_ylabel("Delay (minutes)")
    ax.grid(True, linestyle="--", alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weather_delay_relationships(data: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("temperature", "Temperature (°C)"),
        ("precipitation_mm", "Precipitation (mm)"),
        ("wind_speed_ms", "Wind speed (m/s)"),
        ("relative_humidity", "Relative humidity (%)"),
    ]
    ensure_directory(output_dir)

    for column, label in metrics:
        if column not in data.columns:
            continue
        _scatter_delay_vs_weather(data, column, label, output_dir / f"delay_vs_{column}.png")


def plot_stop_delay_share(
    data: pd.DataFrame, output_dir: Path, top_n: int = 12, min_departures: int = 5, delay_threshold: float = 1.0
) -> None:
    if not {"delay_minutes", "stop_name"}.issubset(data.columns):
        return

    subset = data.dropna(subset=["delay_minutes", "stop_name"])
    if subset.empty:
        return

    stats = (
        subset.assign(is_delayed=lambda df: df["delay_minutes"] > delay_threshold)
        .groupby("stop_name")
        .agg(total=("is_delayed", "size"), delayed=("is_delayed", "sum"))
    )
    stats = stats[stats["total"] >= min_departures]
    if stats.empty:
        return

    stats["delay_share"] = stats["delayed"] / stats["total"]
    stats = stats.sort_values("delay_share", ascending=False).head(top_n)
    stats_to_plot = stats.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    stats_to_plot["delay_share"].plot(kind="barh", ax=ax, color="#6C8E4B")
    ax.set_xlabel("Share of departures with > 1 min delay")
    ax.set_ylabel("Stop")
    ax.set_title("Stops with the highest fraction of delayed departures")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for i, (value, total) in enumerate(zip(stats_to_plot["delay_share"], stats_to_plot["total"])):
        ax.text(value + 0.01, i, f"{value:.0%} ({total} trips)", va="center")

    fig.savefig(output_dir / "stop_delay_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_path(path: Path) -> str:
    """Return a path relative to CWD when possible for nicer logging."""

    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def run_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Generate a set of exploratory plots from the aggregated data."""

    if plt is None:
        print("matplotlib is not installed; skipping plot generation. Run `uv pip install matplotlib` to enable plots.")
        return

    if data.empty:
        print("No data available for plotting.")
        return

    ensure_directory(output_dir)
    plot_mean_delay_by_snapshot(data, output_dir)
    plot_top_stops_by_delay(data, output_dir)
    plot_stop_delay_share(data, output_dir)
    plot_weather_delay_relationships(data, output_dir)
    print(f"Saved plots to {_format_path(output_dir)}.")


def aggregate_departures(export_dir: Path) -> pd.DataFrame:
    """Load and persist a combined departures table."""

    ensure_directory(export_dir)
    aggregated = load_departure_snapshots(export_dir)
    if aggregated.empty:
        print("No departure snapshots found. Place CSVs into the exports directory and re-run.")
        return aggregated

    agg_path = export_dir / AGGREGATED_FILENAME
    aggregated.to_csv(agg_path, index=False)
    print(f"Saved aggregated departures to {_format_path(agg_path)} ({len(aggregated):,} rows).")
    unique_snapshots = aggregated["snapshot_time"].nunique()
    print(f"Loaded {unique_snapshots} snapshot(s) across {aggregated['stop_id'].nunique()} stops.")
    return aggregated


def main() -> None:
    export_dir = EXPORT_DIR.resolve()
    aggregated = aggregate_departures(export_dir)
    if aggregated.empty or not RUN_PLOTS:
        return

    plots_dir = (CUSTOM_PLOTS_DIR or export_dir / PLOTS_SUBDIR).resolve()
    run_plots(aggregated, plots_dir)


if __name__ == "__main__":
    if plt is not None:
        plt.style.use("seaborn-v0_8")  # Nice-looking defaults for quick-look charts.
    main()
