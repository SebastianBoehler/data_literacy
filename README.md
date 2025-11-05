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
