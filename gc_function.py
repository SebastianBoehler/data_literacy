import json
import os
from datetime import datetime
from typing import Dict, Any

import functions_framework
from google.cloud import storage
import pandas as pd

from modules.trias import TriasClient
from modules.utils import (
    attach_weather_to_departures,
    attach_weather_to_stops,
    expand_stops_with_platforms,
    load_config,
    timestamp_slug,
)
from modules.weather import WeatherClient

# Configuration
PROJECT_ID = "data-literacy-477519"
DEFAULT_BUCKET_NAME = "departure_data"
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', DEFAULT_BUCKET_NAME)
DEPARTURE_HORIZON_MINUTES = 60
DEPARTURE_DISCOVERY_MAX_RESULTS = 200
DEPARTURE_MAX_RESULTS_PER_STOP_POINT = 200
CONFIG_PATH = "config.json"


@functions_framework.http
def create_departure_data(request):
    # Validate environment variables
    if not GCS_BUCKET_NAME:
        return json.dumps({
            "error": "GCS_BUCKET_NAME environment variable not set",
            "status": "failed"
        }), 500, {'Content-Type': 'application/json'}
    
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        
        # Initialize clients
        trias = TriasClient(config["trias_requestor_ref"])
        weather_client = WeatherClient()
        storage_client = storage.Client()
        
        # Fetch stops and departures
        center = (config["center_lat"], config["center_lon"])
        stops = trias.fetch_stops(center=center, radius_km=config["search_radius_km"])
        print(f"Fetched {len(stops)} stops within {config['search_radius_km']} km")
        
        discovery_departures = trias.fetch_departures_for_stops(
            stops,
            max_results_per_stop=DEPARTURE_DISCOVERY_MAX_RESULTS,
            horizon_minutes=DEPARTURE_HORIZON_MINUTES,
        )
        discovery_departures = discovery_departures.dropna(subset=["stop_id"])

        # Filter departures by known stops
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

        # Fetch weather data for stops
        weather_by_stop: Dict[str, Dict] = {}
        for _, stop in stops.iterrows():
            lat = stop.get("latitude")
            lon = stop.get("longitude")
            if lat is None or lon is None:
                continue
            
            base_ref = stop.get("trias_ref") or stop.get("stop_id")
            if base_ref not in weather_by_stop:
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
            if pd.notna(stop_point_ref) and stop_point_ref not in weather_by_stop and stop_id in weather_by_stop:
                weather_by_stop[stop_point_ref] = weather_by_stop[stop_id]
        
        # Attach weather data
        stops_with_weather = attach_weather_to_stops(stops_with_platforms, weather_by_stop)
        departures_with_weather = attach_weather_to_departures(departures, weather_by_stop)
        
        # Generate timestamp for filenames
        timestamp = timestamp_slug()
        
        # Save to Google Cloud Storage
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Upload stops data
        stops_blob = bucket.blob(f"stops_{timestamp}.csv")
        stops_csv = stops_with_weather.to_csv(index=False)
        stops_blob.upload_from_string(stops_csv, content_type='text/csv')
        
        # Upload departures data
        departures_blob = bucket.blob(f"departures_{timestamp}.csv")
        departures_csv = departures_with_weather.to_csv(index=False)
        departures_blob.upload_from_string(departures_csv, content_type='text/csv')
        
        # Create metadata file
        metadata = {
            "timestamp": timestamp,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stops_count": len(stops_with_weather),
            "departures_count": len(departures_with_weather),
            "search_radius_km": config["search_radius_km"],
            "center_lat": config["center_lat"],
            "center_lon": config["center_lon"],
            "files": [
                f"gs://{GCS_BUCKET_NAME}/stops_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/departures_{timestamp}.csv"
            ]
        }
        
        metadata_blob = bucket.blob(f"metadata_{timestamp}.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2), 
            content_type='application/json'
        )
        
        response_data = {
            "status": "success",
            "timestamp": timestamp,
            "stops_count": len(stops_with_weather),
            "departures_count": len(departures_with_weather),
            "files_uploaded": [
                f"gs://{GCS_BUCKET_NAME}/stops_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/departures_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/metadata_{timestamp}.json"
            ]
        }
        
        return json.dumps(response_data, indent=2), 200, {'Content-Type': 'application/json'}
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        return json.dumps(error_response, indent=2), 500, {'Content-Type': 'application/json'}


@functions_framework.http
def health_check(request):
    """
    Simple health check endpoint for the Cloud Function.
    """
    return json.dumps({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "bucket": GCS_BUCKET_NAME
    }), 200, {'Content-Type': 'application/json'}
