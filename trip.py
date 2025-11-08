from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from modules.trias import TriasClient
from modules.utils import ensure_directory, load_config, timestamp_slug


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
EXPORT_DIR = SCRIPT_DIR / "exports"
DEPARTURE_LIMIT = 20
DEFAULT_MAX_TRIPS = 50
DEFAULT_DEPARTURE_HORIZON_MINUTES = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch trip-level delay information by retrieving stops, departures, "
            "and detailed journey data from the TRIAS API."
        )
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to configuration JSON containing TRIAS credentials and search parameters.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_DIR),
        help="Directory where CSV exports will be written.",
    )
    parser.add_argument(
        "--departure-limit",
        type=int,
        default=DEPARTURE_LIMIT,
        help="Maximum departures the TRIAS API should return per stop (upper bound).",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=DEFAULT_DEPARTURE_HORIZON_MINUTES,
        help="Time window in minutes for which departures should be retained.",
    )
    parser.add_argument(
        "--max-trips",
        type=int,
        default=DEFAULT_MAX_TRIPS,
        help=(
            "Maximum number of unique journeys (JourneyRef + OperatingDayRef) to request detailed trip info for."
        ),
    )
    return parser.parse_args()


def build_trip_datasets(
    config_path: str,
    output_dir: Path,
    *,
    departure_limit: int,
    horizon_minutes: int,
    max_trips: int | None,
) -> None:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = SCRIPT_DIR / config_file

    config = load_config(str(config_file))

    trias = TriasClient(config["trias_requestor_ref"])
    center = (config["center_lat"], config["center_lon"])

    stops = trias.fetch_stops(center=center, radius_km=config["search_radius_km"])
    print(f"Fetched {len(stops)} stops within {config['search_radius_km']} km")

    departures = trias.fetch_departures_for_stops(
        stops,
        max_results_per_stop=departure_limit,
        horizon_minutes=horizon_minutes,
    )
    departures = departures.dropna(subset=["stop_id"]).reset_index(drop=True)

    known_stop_ids = set(stops["trias_ref"].dropna().unique())
    departures = departures[departures["stop_id"].isin(known_stop_ids)].reset_index(drop=True)
    print(f"Collected {len(departures)} departures across {departures['stop_id'].nunique()} stops")

    unique_lines = (
        departures.dropna(subset=["line_name"])
        .drop_duplicates(subset=["line_name", "destination"])
        .sort_values(["line_name", "destination"])
        .reset_index(drop=True)
    )
    print(f"Identified {len(unique_lines)} unique line/destination combinations")

    trip_calls, trip_positions = trias.fetch_trip_infos_for_departures(
        departures, max_trips=max_trips
    )

    if trip_calls.empty:
        print("No trip call data retrieved. Nothing to export.")
        return

    timestamp = timestamp_slug()
    export_dir = ensure_directory(output_dir)

    calls_export = trip_calls.copy()
    for column in ["arrival_planned", "arrival_estimated", "departure_planned", "departure_estimated"]:
        if column in calls_export.columns and pd.api.types.is_datetime64_any_dtype(calls_export[column]):
            calls_export[column] = calls_export[column].dt.strftime("%Y-%m-%d %H:%M")

    calls_path = export_dir / f"trip_calls_{timestamp}.csv"
    calls_export.to_csv(calls_path, index=False)
    print(f"Saved trip call timeline data to {calls_path}")

    journeys_summary = (
        calls_export.groupby([
            "journey_ref",
            "operating_day_ref",
            "line_name",
            "destination",
        ])
        .agg(
            stops_observed=("stop_sequence", "max"),
            avg_arrival_delay=("arrival_delay_minutes", "mean"),
            max_arrival_delay=("arrival_delay_minutes", "max"),
            avg_departure_delay=("departure_delay_minutes", "mean"),
            max_departure_delay=("departure_delay_minutes", "max"),
        )
        .reset_index()
    )
    journeys_summary[[
        "avg_arrival_delay",
        "max_arrival_delay",
        "avg_departure_delay",
        "max_departure_delay",
    ]] = journeys_summary[[
        "avg_arrival_delay",
        "max_arrival_delay",
        "avg_departure_delay",
        "max_departure_delay",
    ]].round(2)

    summary_path = export_dir / f"trip_summary_{timestamp}.csv"
    journeys_summary.to_csv(summary_path, index=False)
    print(f"Saved trip-level summary metrics to {summary_path}")

    if not trip_positions.empty:
        positions_path = export_dir / f"trip_positions_{timestamp}.csv"
        trip_positions.to_csv(positions_path, index=False)
        print(f"Saved current vehicle position snapshots to {positions_path}")

    lines_path = export_dir / f"lines_{timestamp}.csv"
    unique_lines.to_csv(lines_path, index=False)
    print(f"Saved unique line catalogue to {lines_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    max_trips = args.max_trips if args.max_trips and args.max_trips > 0 else None

    build_trip_datasets(
        config_path=args.config,
        output_dir=output_dir,
        departure_limit=args.departure_limit,
        horizon_minutes=max(args.horizon_minutes, 0) if args.horizon_minutes is not None else DEFAULT_DEPARTURE_HORIZON_MINUTES,
        max_trips=max_trips,
    )


if __name__ == "__main__":
    main()
