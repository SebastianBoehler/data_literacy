# Long Overdue: A Bus Delay Analysis

**Data Literacy Project — University of Tübingen, Winter 2025/26**

## Abstract

By 2025, the common opinion among students and clinicians in Tübingen is that buses are frequently delayed. In order to evaluate this claim, we analyze real-time bus arrival data collected via the TRIAS public transport interface over an 11-week period in winter 2025/26. Results show that delay variability is primarily driven by temporal effects and network structure, with systematic delay accumulation along routes, while weather plays only a secondary role. The schedule change implemented in December 2025 coincides with a sustained reduction in delays.

## 🔗 Links

- **[Interactive Data Exploration](https://sebastianboehler.github.io/data_literacy/)** — Explore delay patterns by line, time period, and weather
- **Paper** — See `paper/` directory for the full report

## Key Findings

- **82.2%** of buses arrive within 3 minutes of schedule
- **~10%** of buses experience delays >5 minutes
- **Schedule change** (Dec 14, 2025) reduced mean delays by **34%**
- **Snow** causes the highest delays (mean 7.5 min vs 2.1 min dry)
- **Delay accumulation**: Each stop adds ~6 seconds of delay on average

## Project Structure

```
├── docs/                      # Interactive GitHub Pages site
├── paper/                     # LaTeX report and figures
├── scripts/                   # Figure generation (fig1, fig2, fig3)
├── modules/                   # TRIAS client, weather API, plot config
├── outputs/                   # Processed data (parquet)
└── docs_generation/           # Scripts to regenerate docs site
```

## Data Collection

Data was collected via the **TRIAS 1.2 SOAP API** (MobiData BW) from November 11, 2025 to January 30, 2026:

- **140,363** trip observations (actual arrivals)
- **366** unique stops, **51** bus lines
- Weather data from **Bright Sky** (DWD observations)

## Reproducing the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Generate paper figures
python scripts/fig1_eda_4panel.py
python scripts/fig2_schedule_change.py
python scripts/fig3_network_graph.py

# Regenerate interactive docs
python docs_generation/generate_all.py
```

## Data Sources

- **[TRIAS API](https://mobidata-bw.de/dataset/trias)** — Real-time public transport data (MobiData BW)
- **[Bright Sky](https://brightsky.dev/)** — DWD weather observations

## License

Built for academic use within the University of Tübingen Data Literacy course (Winter 2025/26).
