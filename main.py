from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.trias import TriasClient
from modules.utils import (
    attach_weather_to_departures,
    attach_weather_to_stops,
    ensure_directory,
    expand_stops_with_platforms,
    load_config,
    timestamp_slug,
)
from modules.weather import WeatherClient

# Re-export Cloud Function handlers for deployment
from gc_function import create_departure_data, health_check

CONFIG_PATH = "config.json"
EXPORT_DIR = Path("exports")
DEPARTURE_HORIZON_MINUTES = 60
DEPARTURE_DISCOVERY_MAX_RESULTS = 200
DEPARTURE_MAX_RESULTS_PER_STOP_POINT = 200


def main() -> None:
    config = load_config(CONFIG_PATH)

    trias = TriasClient(config["trias_requestor_ref"])
    center = (config["center_lat"], config["center_lon"])
    stops = trias.fetch_stops(center=center, radius_km=config["search_radius_km"])
    print(
        f"Fetched {len(stops)} base stops within {config['search_radius_km']} km"
    )

    discovery_departures = trias.fetch_departures_for_stops(
        stops,
        max_results_per_stop=DEPARTURE_DISCOVERY_MAX_RESULTS,
        horizon_minutes=DEPARTURE_HORIZON_MINUTES,
    )
    discovery_departures = discovery_departures.dropna(subset=["stop_id"])

    known_stop_ids = set(stops["trias_ref"].dropna().unique())
    discovery_departures = discovery_departures[
        discovery_departures["stop_id"].isin(known_stop_ids)
    ].reset_index(drop=True)

    stops_with_platforms = expand_stops_with_platforms(stops, discovery_departures)

    departures = trias.fetch_departures_for_stop_points(
        stops_with_platforms,
        max_results_per_stop_point=DEPARTURE_MAX_RESULTS_PER_STOP_POINT,
        horizon_minutes=DEPARTURE_HORIZON_MINUTES,
    )
    if departures.empty:
        departures = discovery_departures.copy()
    departures = departures.dropna(subset=["stop_id"]).reset_index(drop=True)

    platform_coords = (
        stops_with_platforms.dropna(subset=["stop_point_ref"])
        .drop_duplicates(subset=["stop_point_ref"])
        .set_index("stop_point_ref")
    )
    base_coords = stops.set_index("stop_id")

    platform_lat = platform_coords["latitude"] if "latitude" in platform_coords else pd.Series(dtype=float)
    platform_lon = platform_coords["longitude"] if "longitude" in platform_coords else pd.Series(dtype=float)
    base_lat = base_coords["latitude"] if "latitude" in base_coords else pd.Series(dtype=float)
    base_lon = base_coords["longitude"] if "longitude" in base_coords else pd.Series(dtype=float)

    departures["latitude"] = departures["stop_point_ref"].map(platform_lat)
    departures["longitude"] = departures["stop_point_ref"].map(platform_lon)
    departures["latitude"] = departures["latitude"].fillna(departures["stop_id"].map(base_lat))
    departures["longitude"] = departures["longitude"].fillna(departures["stop_id"].map(base_lon))

    weather_client = WeatherClient()
    weather_by_stop: dict[str, dict] = {}
    for _, stop in stops.iterrows():
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue

        base_ref = stop.get("trias_ref") or stop.get("stop_id")
        if base_ref in weather_by_stop:
            weather = weather_by_stop[base_ref]
        else:
            weather = weather_client.fetch_current(lat, lon)

        weather_by_stop[base_ref] = weather
        weather_by_stop[stop["stop_id"]] = weather

    for _, row in stops_with_platforms.iterrows():
        stop_point_ref = row.get("stop_point_ref")
        if pd.notna(stop_point_ref) and stop_point_ref not in weather_by_stop:
            base_ref = row.get("trias_ref") or row.get("stop_id")
            if base_ref in weather_by_stop:
                weather_by_stop[stop_point_ref] = weather_by_stop[base_ref]

    for _, dep in departures.iterrows():
        stop_point_ref = dep.get("stop_point_ref")
        stop_id = dep.get("stop_id")
        if pd.notna(stop_point_ref) and stop_point_ref not in weather_by_stop:
            if stop_id in weather_by_stop:
                weather_by_stop[stop_point_ref] = weather_by_stop[stop_id]

    stops_with_weather = attach_weather_to_stops(stops_with_platforms, weather_by_stop)
    departures_with_weather = attach_weather_to_departures(departures, weather_by_stop)

    ensure_directory(EXPORT_DIR)
    timestamp = timestamp_slug()

    stops_with_weather.to_csv(EXPORT_DIR / f"stops_{timestamp}.csv", index=False)
    departures_with_weather.to_csv(EXPORT_DIR / f"departures_{timestamp}.csv", index=False)

    expanded_count = len(stops_with_platforms)
    print(
        f"Exporting {expanded_count} stops (including platform variants) and "
        f"{len(departures_with_weather)} departures"
    )


if __name__ == "__main__":
    main()
