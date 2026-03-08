# Docs App Structure

This directory contains the static site for the paper companion.

## Entry points

- `index.html`: semantic document shell for the public-facing page
- `methodology.html`: separate documentation page for the pipeline and API notes

## Assets

- `assets/css/index.css`: page-wide styling and responsive layout
- `assets/js/main.js`: bootstrap file
- `assets/js/navigation.js`: sidebar state and section highlighting
- `assets/js/dataExplorer.js`: historical network explorer, plot switching, data table loading
- `assets/js/liveMonitor.js`: supplementary current departure monitor
- `assets/js/config.js`: shared configuration and live stop metadata
- `assets/js/dom.js`: small DOM and formatting helpers

## Data and figures

- `data/`: precomputed JSON summaries used by the explorer
- `lines/`: per-line interactive map HTML exports
- `plots/`: figure images for the paper companion

## Local preview

Serve the folder with a local web server so JSON loading works:

```bash
cd docs
python -m http.server 8000
```
