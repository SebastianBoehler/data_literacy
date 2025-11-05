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

        return {
            "weather_timestamp": weather_timestamp,
            "temperature": weather.get("temperature"),
            "precipitation": weather.get("precipitation"),
            "wind_speed": weather.get("wind_speed"),
            "wind_direction": weather.get("wind_direction"),
            "cloud_cover": weather.get("cloud_cover"),
            "pressure": weather.get("pressure"),
            "relative_humidity": weather.get("relative_humidity"),
            "condition": weather.get("condition"),
            "icon": weather.get("icon"),
        }
