# Data Literacy – Realtime Stops & Weather Snapshot

## Overview

This repository builds a live snapshot of public transport departures in the Tübingen region and enriches each stop/bay with the current weather conditions. It combines the TRIAS 1.2 SOAP interface (MobiData BW) with Bright Sky’s current DWD observations.

The workflow is intentionally lightweight: a single script (`main.py`) orchestrates stop discovery, realtime departure collection, weather lookups, and CSV export.

## Features

- **Stop discovery with bays** – Enumerates every stop place and bay returned by TRIAS within a configurable radius.
- **Realtime departures** – Collects planned vs. estimated departure times, line identifiers, destinations, and platforms for each bay.
- **Current weather join** – Retrieves Bright Sky measurements (temperature, precipitation, wind, cloud cover, pressure, humidity, condition) for the stop coordinates at runtime.
- **CSV exports** – Generates timestamped datasets under `exports/` for further analysis.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure TRIAS access and search radius

Edit `config.json`:

```json
{
  "trias_requestor_ref": "YOUR_REQUESTOR_REF",
  "center_lat": 48.516667,
  "center_lon": 9.05,
  "search_radius_km": 7.0
}
```

- `trias_requestor_ref` – Provided by MobiData BW after access approval.
- `center_lat`, `center_lon` – Coordinates used as the search origin (default: Tübingen city centre).
- `search_radius_km` – Radius in kilometres for stop discovery.

### 3. Run the snapshot

```bash
python main.py
```

Execution prints the number of stops discovered and departures gathered. Fresh CSVs are written to `exports/` (e.g., `stops_YYYYMMDD_HHMMSS.csv`, `departures_YYYYMMDD_HHMMSS.csv`).

## Project Structure

```
code/
├── main.py               # Orchestrates stop discovery, departures, weather, and export
├── modules/
│   ├── trias.py          # TRIAS 1.2 SOAP client (LocationInformation & StopEvent requests)
│   ├── weather.py        # Bright Sky current weather client
│   └── utils.py          # Config loading, timestamp helpers, weather join utilities
├── exports/              # Generated CSV snapshots (timestamped)
└── config.json           # Runtime configuration
```

## TRIAS Module Documentation

The `modules/trias.py` module provides a Python client for the TRIAS 1.2 SOAP interface (MobiData BW). It handles XML request/response parsing and returns pandas DataFrames for easy analysis.

### Initialization

```python
from modules.trias import TriasClient

client = TriasClient(requestor_ref="YOUR_REQUESTOR_REF")
```

| Parameter       | Type                          | Description                           |
| --------------- | ----------------------------- | ------------------------------------- |
| `requestor_ref` | `str`                         | API key provided by MobiData BW       |
| `session`       | `requests.Session` (optional) | Custom session for connection pooling |

### Methods

#### `fetch_stops(center, radius_km, max_results=200)`

Discovers all stops within a geographic radius.

```python
stops = client.fetch_stops(
    center=(48.516667, 9.05),  # (lat, lon)
    radius_km=7.0,
    max_results=200
)
```

**Returns:** DataFrame with columns:

- `stop_id` – Base stop identifier (e.g., `de:08416:11000`)
- `trias_ref` – Full TRIAS reference
- `stop_name` – Human-readable stop name
- `latitude`, `longitude` – Geographic coordinates
- `probability` – TRIAS match probability score

#### `fetch_stop_details(stop_refs)`

Retrieves detailed information for specific stop references.

```python
details = client.fetch_stop_details(["de:08416:11000", "de:08416:11001"])
```

**Returns:** DataFrame with columns:

- `stop_id`, `trias_ref`, `stop_point_ref`, `stop_place_ref`
- `stop_name`, `latitude`, `longitude`

#### `fetch_departures(stop_id, stop_point_ref=None, max_results=200, horizon_minutes=None)`

Fetches real-time departures for a single stop.

```python
departures = client.fetch_departures(
    stop_id="de:08416:11000",
    stop_point_ref="de:08416:11000:11:C",  # Optional: specific platform
    max_results=50,
    horizon_minutes=60  # Optional: filter to next 60 minutes
)
```

**Returns:** DataFrame with columns:

- `stop_id`, `stop_point_ref`, `stop_name`
- `planned_time`, `estimated_time` – Scheduled vs. real-time departure
- `delay_minutes` – Calculated delay (estimated - planned)
- `line_name` – Bus/train line identifier (e.g., "1", "N91")
- `destination` – Final destination of the service
- `platform` – Platform/bay identifier
- `journey_ref`, `operating_day_ref` – Unique trip identifiers

#### `fetch_departures_for_stop_points(stops, max_results_per_stop_point=200, horizon_minutes=None)`

Batch fetches departures for multiple stop points from a stops DataFrame.

```python
all_departures = client.fetch_departures_for_stop_points(
    stops=stops_df,  # Must contain 'stop_point_ref' column
    max_results_per_stop_point=100,
    horizon_minutes=120
)
```

#### `fetch_departures_for_stops(stops, max_results_per_stop=200, max_stops=None, horizon_minutes=None)`

Batch fetches departures for multiple stops (by stop_id).

```python
all_departures = client.fetch_departures_for_stops(
    stops=stops_df,  # Must contain 'stop_id' column
    max_results_per_stop=100,
    max_stops=50  # Optional: limit number of stops
)
```

#### `fetch_trip_info(journey_ref, operating_day_ref=None, **options)`

Retrieves detailed trip information including all stops along the route.

```python
trip = client.fetch_trip_info(
    journey_ref="tub:09001::H:j25:1",
    operating_day_ref="2025-11-15",
    include_calls=True,
    include_estimated=True,
    include_position=True
)
```

**Returns:** Dictionary with:

- `calls` – DataFrame of all stops on the trip with arrival/departure times
- `service` – Service metadata (line_name, destination, journey_ref)
- `current_position` – Current vehicle position (lat, lon, bearing) if available

**Calls DataFrame columns:**

- `phase` – "previous" (already passed) or "onward" (upcoming)
- `stop_point_ref`, `stop_name`, `stop_sequence`, `platform`
- `arrival_planned`, `arrival_estimated`, `arrival_delay_minutes`
- `departure_planned`, `departure_estimated`, `departure_delay_minutes`

#### `fetch_trip_infos_for_departures(departures, max_trips=None)`

Batch fetches trip info for all unique journeys in a departures DataFrame.

```python
calls_df, positions_df = client.fetch_trip_infos_for_departures(
    departures=departures_df,
    max_trips=100  # Optional: limit API calls
)
```

**Returns:** Tuple of (calls_df, positions_df)

### Time Handling

All timestamps are automatically converted from UTC to Europe/Berlin timezone and returned as timezone-naive datetime objects for compatibility with pandas operations.

### Error Handling

The client raises `RuntimeError` for empty TRIAS responses and `requests.HTTPError` for HTTP failures. API responses are logged to stdout with status codes and body previews for debugging.

## Exports

- **`stops_*.csv`** – One row per stop/bay returned by TRIAS within the configured radius. Includes unique `stop_id`, original TRIAS reference, coordinates, and probability.
- **`departures_*.csv`** – Realtime departures per stop/bay with line, destination, platform, delay (minutes), and attached weather measurements (temperature, precipitation, wind, cloud cover, pressure, humidity, condition).

## Notes on weather metrics

Bright Sky provides precipitation and wind values across several trailing intervals (10 / 30 / 60 minutes). The code surfaces the most recent available measurement to represent current conditions while keeping the original interval columns in the CSV for scientific interpretation.

## Data Sources

1. **TRIAS 1.2 SOAP (MobiData BW)** – Stop discovery (`LocationInformationRequest`) and realtime departures (`StopEventRequest`).
2. **Bright Sky (DWD)** – Current weather observations via `https://api.brightsky.dev/current_weather`.

## License & Usage

- Built for academic use within the University of Tübingen Data Literacy course.
- Respect MobiData BW’s API terms, Bright Sky usage guidelines, and any applicable rate limits.
