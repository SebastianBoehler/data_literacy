# Data Collection and Analysis Summary

This document provides a comprehensive summary of the data collection methodology, data sources, APIs, structure, and analysis methods used in this Data Literacy project on public transit delays in Tübingen.

---

## 1. Data Sources

### 1.1 TRIAS 1.2 SOAP API (MobiData BW)

**Endpoint:** `https://efa-bw.de/trias`

**Purpose:** Real-time public transit data for Baden-Württemberg

**Authentication:** Requestor reference key provided by MobiData BW

**Request Types Used:**

| Request Type                 | Purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `LocationInformationRequest` | Stop discovery within geographic radius               |
| `StopEventRequest`           | Real-time departure information per stop              |
| `TripInfoRequest`            | Detailed journey/trip information with stop sequences |

**Key Parameters:**

- XML namespace: `http://www.vdv.de/trias` (TRIAS 1.2 standard)
- SIRI namespace: `http://www.siri.org.uk/siri`
- Request timeout: 30 seconds
- Real-time data: `IncludeRealtimeData=true`

### 1.2 Bright Sky API (DWD - Deutscher Wetterdienst)

**Endpoint:** `https://api.brightsky.dev/current_weather`

**Purpose:** Current weather observations from German Weather Service stations

**Authentication:** None required (open API)

**Weather Variables Collected:**

- `temperature` (°C)
- `precipitation_mm` (10/30/60 min intervals)
- `wind_speed_ms` (10/30/60 min intervals)
- `wind_direction_deg`
- `cloud_cover` (%)
- `pressure_hpa` (mean sea level)
- `relative_humidity` (%)
- `condition` (categorical: dry, rain, snow, etc.)
- `weather_station_name` (source station identifier)

**Weather Station:** Rottenburg-Kiebingen (nearest DWD station to Tübingen)

### 1.3 Google Cloud Storage

**Bucket:** `departure_data`

**Purpose:** Persistent storage for timestamped data snapshots

**File Types Stored:**

- `departures_YYYYMMDD_HHMMSS.csv`
- `stops_YYYYMMDD_HHMMSS.csv`
- `trip_calls_YYYYMMDD_HHMMSS.csv`
- `trip_summary_YYYYMMDD_HHMMSS.csv`
- `trip_positions_YYYYMMDD_HHMMSS.csv`
- `lines_YYYYMMDD_HHMMSS.csv`
- `metadata_YYYYMMDD_HHMMSS.json`

---

## 2. Data Collection Architecture

### 2.1 Google Cloud Function (`gc_function.py`)

A serverless HTTP-triggered function deployed on Google Cloud Platform that executes the data collection pipeline.

**Execution Flow:**

1. Load configuration (center coordinates, search radius)
2. Initialize TRIAS and Weather clients
3. Fetch stops within radius via `LocationInformationRequest`
4. Fetch departures for discovered stops via `StopEventRequest`
5. Expand stops with platform/bay information
6. Fetch detailed trip information via `TripInfoRequest`
7. Attach weather data to each stop/departure
8. Upload all datasets to GCS with timestamps

**Configuration Parameters:**

```json
{
  "trias_requestor_ref": "SeBaSTiaN_BoeHLeR",
  "center_lat": 48.516667,
  "center_lon": 9.05,
  "search_radius_km": 7.0
}
```

**Collection Parameters:**

- `DEPARTURE_HORIZON_MINUTES`: 60 (look-ahead window)
- `DEPARTURE_DISCOVERY_MAX_RESULTS`: 200
- `DEPARTURE_MAX_RESULTS_PER_STOP_POINT`: 200
- `TRIP_MAX_TRIPS`: 100

### 2.2 TRIAS Client (`trias.py`)

Custom Python client implementing TRIAS 1.2 SOAP protocol.

**Key Methods:**

| Method                               | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `fetch_stops()`                      | Discover stops within geographic circle     |
| `fetch_departures()`                 | Get real-time departures for a single stop  |
| `fetch_departures_for_stops()`       | Batch departures for multiple stops         |
| `fetch_departures_for_stop_points()` | Platform-level departure queries            |
| `fetch_trip_info()`                  | Detailed journey information with all calls |
| `fetch_trip_infos_for_departures()`  | Batch trip info for departures              |
| `fetch_stop_details()`               | Detailed stop/platform information          |

**Delay Calculation:**

```python
delay_minutes = (estimated_time - planned_time).total_seconds() / 60
```

**Timezone Handling:** All times converted from UTC to `Europe/Berlin` local time.

---

## 2.3 Trip Data Collection and Deduplication

The trip data collection process retrieves detailed stop-by-stop journey information for each departure, enabling delay propagation analysis along bus routes.

### Deduplication Strategy

Since the same bus journey appears at multiple stops within the search radius, departures are deduplicated before fetching trip info:

```python
unique = (
    departures.dropna(subset=["journey_ref"])
    .drop_duplicates(subset=["journey_ref", "operating_day_ref"])
)
```

**Key:** `(journey_ref, operating_day_ref)` uniquely identifies a single bus journey on a specific day.

### Trip Info Request

For each unique journey, a `TripInfoRequest` is sent to TRIAS, which returns:

1. **Service metadata:** Line name, destination, journey reference
2. **Current position:** Real-time GPS coordinates and bearing (if available)
3. **Call sequence:** All stops the bus visits, with arrival/departure times

### Call Parsing and Phase Classification

Each stop in a journey is classified by **phase**:

| Phase      | Meaning                                |
| ---------- | -------------------------------------- |
| `previous` | Stops already visited (bus has passed) |
| `onward`   | Upcoming stops (bus will visit)        |

The TRIAS response contains `<PreviousCall>` and `<OnwardCall>` XML elements, parsed into a unified DataFrame.

### Stop Sequence Structure

Each call record contains:

| Field                     | Description                           |
| ------------------------- | ------------------------------------- |
| `stop_sequence`           | Order in trip (1, 2, 3, ...)          |
| `stop_point_ref`          | Platform-level stop ID                |
| `stop_name`               | Human-readable name                   |
| `arrival_planned`         | Scheduled arrival time                |
| `arrival_estimated`       | Real-time arrival prediction          |
| `departure_planned`       | Scheduled departure time              |
| `departure_estimated`     | Real-time departure prediction        |
| `arrival_delay_minutes`   | Arrival delay (estimated - planned)   |
| `departure_delay_minutes` | Departure delay (estimated - planned) |

### Example: Bus Line 91 Journey

```
stop_sequence | stop_name                    | phase    | departure_delay
1             | Tübingen Hauptbahnhof        | previous | 5.2 min
2             | Tübingen Neckarbrücke        | onward   | 5.2 min
3             | Tübingen Wilhelmstraße       | onward   | 5.2 min
4             | Tübingen Uni / Neue Aula     | onward   | 5.2 min
...           | ...                          | ...      | ...
```

This structure enables analysis of:

- **Delay propagation:** How delays accumulate or dissipate along a route
- **Stop-to-stop transitions:** Which segments cause delays
- **Recovery patterns:** Where buses catch up to schedule

### Sorting and Output

Final trip calls DataFrame is sorted by:

```python
calls_df.sort_values(by=["journey_ref", "stop_sequence", "phase"])
```

This ensures chronological order within each journey for time-series analysis.

---

## 3. Data Structure

### 3.1 Departures Dataset

**Total Records:** ~210,139 rows (1,268 collection snapshots)

**Date Range:** 2025-11-11 to 2026-01-12

**Columns (26 total):**

| Column                 | Type     | Description                                   |
| ---------------------- | -------- | --------------------------------------------- |
| `stop_id`              | string   | Base stop identifier (e.g., `de:08416:11000`) |
| `stop_point_ref`       | string   | Platform-level reference                      |
| `stop_name`            | string   | Human-readable stop name                      |
| `planned_time`         | datetime | Scheduled departure time                      |
| `estimated_time`       | datetime | Real-time estimated departure                 |
| `delay_minutes`        | float    | Difference: estimated - planned               |
| `line_name`            | string   | Bus/train line identifier                     |
| `destination`          | string   | Route destination                             |
| `platform`             | string   | Platform/bay designation                      |
| `journey_ref`          | string   | Unique journey identifier                     |
| `operating_day_ref`    | date     | Operating day reference                       |
| `latitude`             | float    | Stop latitude                                 |
| `longitude`            | float    | Stop longitude                                |
| `temperature`          | float    | Temperature (°C)                              |
| `precipitation_mm`     | float    | Precipitation amount                          |
| `wind_speed_ms`        | float    | Wind speed (m/s)                              |
| `wind_direction_deg`   | float    | Wind direction (degrees)                      |
| `cloud_cover`          | float    | Cloud cover (%)                               |
| `pressure_hpa`         | float    | Atmospheric pressure                          |
| `relative_humidity`    | float    | Relative humidity (%)                         |
| `condition`            | string   | Weather condition category                    |
| `weather_station_name` | string   | Source weather station                        |
| `timestamp`            | datetime | Data collection timestamp                     |

### 3.2 Trip Calls Dataset

**Total Records:** ~1,631,727 rows

**Purpose:** Stop-by-stop journey information for delay propagation analysis

**Key Columns:**

- `journey_ref`, `operating_day_ref`
- `stop_sequence` (order in trip)
- `phase` (previous/onward)
- `arrival_planned`, `arrival_estimated`
- `departure_planned`, `departure_estimated`
- `arrival_delay_minutes`, `departure_delay_minutes`
- `line_name`, `destination`

---

## 4. Data Filtering and Preprocessing

### 4.1 Schedule Change Filter

The city of Tübingen changed bus schedules on **December 14, 2025**. The notebook supports flags to sample over all, pre or post data to see changes in delay patterns.

```python
SCHEDULE_CHANGE_DATE = pd.Timestamp('2025-12-14')
all_departure_data = all_departure_data[all_departure_data['timestamp'] >= SCHEDULE_CHANGE_DATE]
```

**Filtered Dataset:** 96,052 rows (post-schedule change)

### 4.2 Outlier Removal

Extreme delay values (>90 minutes) identified as data errors and removed.

```python
DELAY_MAX_THRESHOLD = 90  # minutes
outlier_mask = all_departure_data['delay_minutes'] > DELAY_MAX_THRESHOLD
# Removed: 4 outliers (all from line 7625 at Tübingen Poststraße)
```

### 4.3 Derived Variables

| Variable        | Derivation                                                |
| --------------- | --------------------------------------------------------- |
| `hour`          | Extracted from `planned_time`                             |
| `weekday`       | Day name from `timestamp`                                 |
| `is_weekend`    | Boolean: Saturday or Sunday                               |
| `is_late`       | Boolean: `delay_minutes > 0`                              |
| `delay_pos`     | Delay only for late buses                                 |
| `is_rainy`      | Boolean: `precipitation_mm > 0`                           |
| `precip_bin`    | Categorical: 0, 0-0.5, 0.5-2, 2-10, 10+ mm                |
| `daypart`       | Categorical: Nacht, Morgen-Peak, Mittag, Abend-Peak, Spät |
| `stop_busyness` | Departure count per stop (traffic proxy)                  |

---

## 5. Statistical Methods

### 5.1 Bootstrap Confidence Intervals

Used throughout for robust uncertainty quantification.

```python
def bootstrap_ci(x, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats[i] = stat_fn(sample)
    alpha = 1 - ci
    return (np.quantile(stats, alpha / 2), np.quantile(stats, 1 - alpha / 2))
```

**Parameters:**

- Bootstrap iterations: 1,000-2,000
- Confidence level: 95%
- Applied to: mean, median

### 5.2 Group Comparisons

Groups compared against overall baseline using CI non-overlap criterion:

- **Clearly higher:** Group CI lower bound > Overall CI upper bound
- **Clearly lower:** Group CI upper bound < Overall CI lower bound
- **Overlap:** CIs intersect (no significant difference)

### 5.3 Minimum Sample Size

Groups with n < 50-200 (depending on analysis) excluded for statistical reliability.

---

## 6. Analysis Dimensions (Hypotheses)

| Hypothesis           | Grouping Variable               | Key Finding                                        |
| -------------------- | ------------------------------- | -------------------------------------------------- |
| H1: Buses often late | Overall                         | Mean delay: 0.43 min, ~14% late                    |
| H2: Weather impact   | `condition`, `precipitation_mm` | Rain increases delay (+0.36 min)                   |
| H3: Stop dependence  | `stop_name`                     | Range: -0.11 to +1.03 min by stop                  |
| H4: Time of day      | `hour`, `daypart`               | Peak at 17:00 (0.70 min), lowest 23:00 (0.18 min)  |
| H5: Line dependence  | `line_name`                     | RE 6 highest (4.28 min), Line 13 lowest (0.03 min) |
| H6: Temperature      | `temperature`                   | Mild temps (10-20°C) show higher delays            |
| H8: Weekend effect   | `is_weekend`                    | Weekdays: 0.47 min, Weekends: 0.31 min             |

---

## 7. Visualization Methods

### 7.1 Libraries Used

- **matplotlib** (v3.x): Publication-quality static figures
- **seaborn**: Statistical visualization
- **plotly**: Interactive visualizations
- **networkx**: Transit network graph construction
- **ipysigma**: Interactive network visualization

### 7.2 Figure Types

| Figure             | Purpose                                                               |
| ------------------ | --------------------------------------------------------------------- |
| 4-panel EDA        | Delay distribution, hourly pattern, busiest stops, weather conditions |
| Hypothesis figures | Per-hypothesis bar charts with bootstrap CIs                          |
| Network graph      | Transit network topology visualization                                |
| Calendar heatmap   | Temporal delay patterns                                               |
| Geographic map     | Stop locations with delay coloring                                    |

### 7.3 Academic Style Settings

```python
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})
```

---

## 8. Data Loading Pipeline

### 8.1 Parallel Download from GCS

```python
def download_and_combine_data(file_list, max_workers=20):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Parallel download of CSV files
        # Extract timestamp from filename pattern: YYYYMMDD_HHMMSS
        # Combine into single DataFrame
```

**Performance:** 1,268 files processed in parallel with 20 workers

---

## 9. Key Technical Choices

| Choice                      | Rationale                                          |
| --------------------------- | -------------------------------------------------- |
| 7 km search radius          | Covers Tübingen urban area + surrounding villages  |
| 60-min departure horizon    | Captures near-term real-time predictions           |
| Platform-level granularity  | More precise delay attribution                     |
| Bootstrap CIs               | Non-parametric, robust to non-normal distributions |
| Post-schedule-change filter | Ensures data consistency after network changes     |
| 90-min outlier threshold    | Removes implausible data errors                    |

---

## 10. Output Files

| File                                      | Description                      |
| ----------------------------------------- | -------------------------------- |
| `eda_4panel.pdf/png`                      | Exploratory data analysis figure |
| `fig3_hypothesis3_stops.pdf/png`          | Stop-level delay analysis        |
| `fig4_hypothesis4_hourly.pdf/png`         | Hourly delay patterns            |
| `fig5_hypothesis5_lines.pdf/png`          | Line-level delay comparison      |
| `grunddaten_mean_median_ci95_summary.csv` | Summary statistics table         |
| `hypothesis_summary.csv`                  | Hypothesis test results          |

---

## 11. Dependencies

```
google-cloud-storage
pandas
plotly
networkx
ipysigma
requests
functions-framework
matplotlib
seaborn
numpy
```

---

## 12. Data Quality Notes

- **Missing estimated times:** Some departures lack real-time predictions (NaN delay)
- **Weather station coverage:** Single station (Rottenburg-Kiebingen) for entire area
- **Platform identification:** Not all stops have platform-level data
- **Schedule change:** Data before Dec 14, 2025 uses different route structure
