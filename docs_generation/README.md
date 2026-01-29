# Documentation Generation

Scripts for generating the interactive GitHub Pages documentation.

## Quick Start

Generate all documentation artifacts with a single command:

```bash
python docs_generation/generate_all.py
```

Or from within the `docs_generation/` directory:

```bash
python generate_all.py
```

## Prerequisites

1. Run the Jupyter notebook to export `outputs/all_trip_data.parquet`
2. Ensure required packages are installed: `pandas`, `numpy`, `networkx`, `plotly`, `matplotlib`

## Scripts

| Script | Description |
|--------|-------------|
| `generate_all.py` | Main entry point - runs all generation scripts |
| `network_graphs.py` | Generates per-line network maps with delay coloring |
| `eda_plots.py` | Generates EDA plots (distributions, hourly patterns, etc.) |

## Output Structure

All artifacts are generated in `docs/` with period-specific subdirectories:

```
docs/
├── index.html              # Main page with period toggle
├── lines/
│   ├── all/               # Network maps (all data)
│   ├── pre/               # Network maps (before Dec 14, 2025)
│   └── post/              # Network maps (after Dec 14, 2025)
├── data/
│   ├── all/               # JSON data (all data)
│   ├── pre/               # JSON data (before schedule change)
│   └── post/              # JSON data (after schedule change)
└── plots/
    ├── all/               # EDA plots (all data)
    ├── pre/               # EDA plots (before schedule change)
    └── post/              # EDA plots (after schedule change)
```

## Time Periods

| Period | Description | Date Filter |
|--------|-------------|-------------|
| `all` | All data | None |
| `pre` | Before schedule change | < 2025-12-14 |
| `post` | After schedule change | >= 2025-12-14 |

## Adding New Plots

1. Create a new script in `docs_generation/` (e.g., `new_plots.py`)
2. Follow the pattern of `eda_plots.py` with `PERIODS` and `filter_by_period()`
3. Add the script to `generate_all.py` in the `scripts` list
