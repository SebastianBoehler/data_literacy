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
TRIP_MAX_TRIPS = 100
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

        unique_lines = (
            departures.dropna(subset=["line_name"])
            .drop_duplicates(subset=["line_name", "destination"])
            .sort_values(["line_name", "destination"])
            .reset_index(drop=True)
        )

        trip_calls, trip_positions = trias.fetch_trip_infos_for_departures(
            departures, max_trips=TRIP_MAX_TRIPS
        )

        trip_exports: Dict[str, pd.DataFrame] = {}
        trip_counts = {
            "trip_calls_rows": 0,
            "trip_summary_rows": 0,
            "trip_positions_rows": 0,
            "lines_rows": 0,
        }

        if not trip_calls.empty:
            calls_export = trip_calls.copy()
            datetime_columns = [
                "arrival_planned",
                "arrival_estimated",
                "departure_planned",
                "departure_estimated",
            ]
            for column in datetime_columns:
                if column in calls_export.columns and pd.api.types.is_datetime64_any_dtype(
                    calls_export[column]
                ):
                    calls_export[column] = calls_export[column].dt.strftime("%Y-%m-%d %H:%M:%S")

            journeys_summary = (
                calls_export.groupby(
                    [
                        "journey_ref",
                        "operating_day_ref",
                        "line_name",
                        "destination",
                    ]
                )
                .agg(
                    stops_observed=("stop_sequence", "max"),
                    avg_arrival_delay=("arrival_delay_minutes", "mean"),
                    max_arrival_delay=("arrival_delay_minutes", "max"),
                    avg_departure_delay=("departure_delay_minutes", "mean"),
                    max_departure_delay=("departure_delay_minutes", "max"),
                )
                .reset_index()
            )

            delay_columns = [
                "avg_arrival_delay",
                "max_arrival_delay",
                "avg_departure_delay",
                "max_departure_delay",
            ]
            existing_delay_columns = [col for col in delay_columns if col in journeys_summary.columns]
            if existing_delay_columns:
                journeys_summary[existing_delay_columns] = journeys_summary[
                    existing_delay_columns
                ].round(2)

            trip_exports["trip_calls"] = calls_export
            trip_exports["trip_summary"] = journeys_summary
            trip_counts["trip_calls_rows"] = len(calls_export)
            trip_counts["trip_summary_rows"] = len(journeys_summary)
        else:
            journeys_summary = pd.DataFrame()

        if not trip_positions.empty:
            trip_exports["trip_positions"] = trip_positions
            trip_counts["trip_positions_rows"] = len(trip_positions)

        if not unique_lines.empty:
            trip_exports["lines"] = unique_lines
            trip_counts["lines_rows"] = len(unique_lines)

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

        trip_file_prefixes = {
            "trip_calls": "trip_calls",
            "trip_summary": "trip_summary",
            "trip_positions": "trip_positions",
            "lines": "lines",
        }

        trip_files_uploaded: list[str] = []
        for key, df in trip_exports.items():
            blob_name = f"{trip_file_prefixes[key]}_{timestamp}.csv"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
            trip_files_uploaded.append(f"gs://{GCS_BUCKET_NAME}/{blob_name}")

        # Create metadata file
        metadata = {
            "timestamp": timestamp,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stops_count": len(stops_with_weather),
            "departures_count": len(departures_with_weather),
            "search_radius_km": config["search_radius_km"],
            "center_lat": config["center_lat"],
            "center_lon": config["center_lon"],
            "trip_stats": trip_counts,
            "files": [
                f"gs://{GCS_BUCKET_NAME}/stops_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/departures_{timestamp}.csv"
            ]
        }

        metadata["files"].extend(trip_files_uploaded)

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
            "trip_stats": trip_counts,
            "files_uploaded": [
                f"gs://{GCS_BUCKET_NAME}/stops_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/departures_{timestamp}.csv",
                f"gs://{GCS_BUCKET_NAME}/metadata_{timestamp}.json",
                *trip_files_uploaded,
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
