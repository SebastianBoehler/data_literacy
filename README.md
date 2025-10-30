# Data Literacy

## Overview

Analyzes public transport punctuality in Tübingen region in relation to weather conditions by integrating real-time transport data from EFA-BW with weather data.

## Features

- **Dynamic Stop Discovery**: Automatically discovers all Tübingen transport stops (238 unique stops)
- **Real-time Data Collection**: Fetches live departure data using EFA-BW Haltestellenmonitor
- **Weather Integration**: Collects current weather conditions for each stop location

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Process all Tübingen stops
python data_integration_pipeline.py

# Test with limited stops (faster)
python data_integration_pipeline.py --limit 10

# Use predefined sample stops
python data_integration_pipeline.py --sample
```

## Project Structure

```
code/
├── data_integration_pipeline.py    # Main pipeline
├── modules/                         # Core modules
│   ├── realtime_data.py            # Transport data fetching
│   ├── weather_data.py             # Weather data collection
│   └── analysis.py                 # Data analysis
├── exports/                         # Generated datasets
└── docs/                           # Documentation
```

## Output Files

Generated in `exports/` directory:

- `all_dataset_*.csv` - Main integrated dataset (transport + weather)
- `raw_realtime_data_*.csv` - Raw transport departure data
- `current_weather_data_*.csv` - Weather data for all stops

## Data Sources

1. **GTFS-API (MobiData BW)** - Static transport data and stop information
2. **EFA-BW Haltestellenmonitor** - Real-time departure data
3. **Bright Sky (DWD)** - Weather observations and forecasts

## License

Academic use for University of Tübingen Data Literacy course. All data sources used according to their respective terms.
