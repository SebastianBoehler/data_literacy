from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.trias import TriasClient
from modules.utils import (
    attach_weather_to_departures,
    attach_weather_to_stops,
    ensure_directory,
    load_config,
    timestamp_slug,
)
from modules.weather import WeatherClient

CONFIG_PATH = "config.json"
EXPORT_DIR = Path("exports")
DEPARTURE_LIMIT = 10


def main() -> None:
    config = load_config(CONFIG_PATH)

    trias = TriasClient(config["trias_requestor_ref"])
    center = (config["center_lat"], config["center_lon"])
    stops = trias.fetch_stops(center=center, radius_km=config["search_radius_km"])
    print(f"Fetched {len(stops)} stops within {config['search_radius_km']} km")

    departures = trias.fetch_departures_for_stops(stops, limit_per_stop=DEPARTURE_LIMIT)
    departures = departures.dropna(subset=["stop_id"]).reset_index(drop=True)

    weather_client = WeatherClient()
    weather_by_stop: dict[str, dict] = {}
    for _, stop in stops.iterrows():
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        weather_by_stop[stop["stop_id"]] = weather_client.fetch_current(lat, lon)

    stops_with_weather = attach_weather_to_stops(stops, weather_by_stop)
    departures_with_weather = attach_weather_to_departures(departures, weather_by_stop)

    ensure_directory(EXPORT_DIR)
    timestamp = timestamp_slug()

    stops_with_weather.to_csv(EXPORT_DIR / f"stops_{timestamp}.csv", index=False)
    departures_with_weather.to_csv(EXPORT_DIR / f"departures_{timestamp}.csv", index=False)

    print(
        f"Stops: {len(stops_with_weather)}, departures: {len(departures_with_weather)}"
    )


if __name__ == "__main__":
    main()
