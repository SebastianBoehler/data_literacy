from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
from zoneinfo import ZoneInfo

WEATHER_CURRENT_URL = "https://api.brightsky.dev/current_weather"
LOCAL_TZ = ZoneInfo("Europe/Berlin")


class WeatherClient:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch_current(self, lat: float, lon: float) -> dict:
        params = {"lat": lat, "lon": lon}
        response = self.session.get(WEATHER_CURRENT_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        weather = payload.get("weather")
        if not weather:
            return {}

        timestamp_raw = weather.get("timestamp")
        weather_timestamp = None
        if timestamp_raw:
            try:
                aware_utc = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                weather_timestamp = aware_utc.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                weather_timestamp = timestamp_raw

        def coalesce(*keys: str) -> Optional[float]:
            for key in keys:
                if key is None:
                    continue
                value = weather.get(key)
                if value is not None:
                    return value
            return None

        # Bright Sky exposes precipitation/wind metrics over multiple trailing intervals (10/30/60 min)
        # and recently added aggregate keys (without suffix). We pick the freshest available value.

        precip = coalesce("precipitation_60", "precipitation_30", "precipitation_10", "precipitation")

        wind_speed = coalesce("wind_speed_10", "wind_speed_30", "wind_speed_60", "wind_speed")

        wind_direction = coalesce(
            "wind_direction_10",
            "wind_direction_30",
            "wind_direction_60",
            "wind_direction",
        )

        pressure = coalesce("pressure_msl", "pressure")

        cloud_cover = coalesce("cloud_cover", "cloud_cover_total")

        source_id = weather.get("source_id")
        station_name: Optional[str] = None
        sources = payload.get("sources") or []
        if source_id is not None:
            for source in sources:
                if source.get("id") == source_id:
                    station_name = source.get("station_name") or source.get("dwd_station_id")
                    break
        if station_name is None and sources:
            station_name = sources[0].get("station_name") or sources[0].get("dwd_station_id")

        return {
            "weather_timestamp": weather_timestamp,
            "temperature": weather.get("temperature"),
            "precipitation_mm": precip,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_direction,
            "cloud_cover": cloud_cover,
            "pressure_hpa": pressure,
            "relative_humidity": weather.get("relative_humidity"),
            "condition": weather.get("condition"),
            "icon": weather.get("icon"),
            "weather_source_id": source_id,
            "weather_station_name": station_name,
        }
