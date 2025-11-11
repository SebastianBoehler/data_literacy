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
        key = None
        if "stop_point_ref" in row and pd.notna(row["stop_point_ref"]):
            key = row["stop_point_ref"]
        elif "trias_ref" in row and pd.notna(row["trias_ref"]):
            key = row["trias_ref"]
        else:
            key = row["stop_id"]
        weather_rows.append(weather_by_stop.get(key, {}))

    weather_df = pd.DataFrame(weather_rows)
    for col, default in (
        ("precipitation_mm", 0.0),
        ("wind_speed_ms", 0.0),
        ("wind_direction_deg", None),
        ("pressure_hpa", None),
    ):
        if col not in weather_df.columns:
            weather_df[col] = default
    return pd.concat([enriched.reset_index(drop=True), weather_df], axis=1)


def expand_stops_with_platforms(stops: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    if stops.empty:
        return stops

    base = stops.copy()
    base["stop_point_ref"] = base["stop_id"]
    base["platform"] = pd.NA

    if departures.empty or "stop_point_ref" not in departures.columns:
        return base

    stop_cols = list(stops.columns)
    platform_records = (
        departures.dropna(subset=["stop_point_ref"])
        .drop_duplicates(subset=["stop_point_ref"])
        .loc[:, ["stop_id", "stop_point_ref", "platform"]]
    )

    if platform_records.empty:
        return base

    merged = platform_records.merge(stops, on="stop_id", how="left")

    missing_stop_mask = merged["stop_name"].isna()
    if missing_stop_mask.any():
        fallback = stops.set_index("stop_id")
        for idx in merged.index[missing_stop_mask]:
            stop_id = merged.at[idx, "stop_id"]
            if stop_id in fallback.index:
                for col in stop_cols:
                    merged.at[idx, col] = fallback.at[stop_id, col]

    platform_only = merged[merged["stop_point_ref"] != merged["stop_id"]]

    return pd.concat([base, platform_only], ignore_index=True, sort=False)


def attach_weather_to_departures(
    departures: pd.DataFrame, weather_by_stop: dict[str, dict]
) -> pd.DataFrame:
    if departures.empty:
        return departures

    enriched = departures.copy()
    weather_rows = []
    for _, row in enriched.iterrows():
        key = None
        if "stop_point_ref" in row and pd.notna(row["stop_point_ref"]):
            key = row["stop_point_ref"]
        elif "trias_ref" in row and pd.notna(row["trias_ref"]):
            key = row["trias_ref"]
        else:
            key = row["stop_id"]
        weather_rows.append(weather_by_stop.get(key, {}))

    weather_df = pd.DataFrame(weather_rows)
    for col, default in (
        ("precipitation_mm", 0.0),
        ("wind_speed_ms", 0.0),
        ("wind_direction_deg", None),
        ("pressure_hpa", None),
    ):
        if col not in weather_df.columns:
            weather_df[col] = default
    merged = pd.concat([enriched.reset_index(drop=True), weather_df], axis=1)

    return merged
