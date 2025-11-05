import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REQUIRED_CONFIG_KEYS = {"trias_requestor_ref", "center_lat", "center_lon", "search_radius_km"}


def load_config(config_path: str) -> Dict[str, Any]:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    missing = REQUIRED_CONFIG_KEYS.difference(data)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Config missing required keys: {missing_keys}")
    return data


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_slug() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def attach_weather_to_stops(stops: pd.DataFrame, weather_by_stop: dict[str, dict]) -> pd.DataFrame:
    if stops.empty:
        return stops

    enriched = stops.copy()
    weather_rows = []
    for _, row in enriched.iterrows():
        weather_rows.append(weather_by_stop.get(row["stop_id"], {}))

    weather_df = pd.DataFrame(weather_rows)
    return pd.concat([enriched.reset_index(drop=True), weather_df], axis=1)


def attach_weather_to_departures(
    departures: pd.DataFrame, weather_by_stop: dict[str, dict]
) -> pd.DataFrame:
    if departures.empty:
        return departures

    enriched = departures.copy()
    weather_rows = []
    for _, row in enriched.iterrows():
        weather_rows.append(weather_by_stop.get(row["stop_id"], {}))

    weather_df = pd.DataFrame(weather_rows)
    merged = pd.concat([enriched.reset_index(drop=True), weather_df], axis=1)

    if "planned_time" in merged.columns:
        merged["planned_time"] = merged["planned_time"].dt.strftime("%Y-%m-%d %H:%M")
    if "estimated_time" in merged.columns:
        merged["estimated_time"] = merged["estimated_time"].dt.strftime("%Y-%m-%d %H:%M")
    return merged
