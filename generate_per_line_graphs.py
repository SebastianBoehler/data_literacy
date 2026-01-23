#!/usr/bin/env python3
"""
Generate per-line network graphs for GitHub Pages visualization.
Uses all_trip_data from the notebook (exported as parquet) for accurate results.

Usage:
1. Run the notebook cells to generate all_trip_data
2. Export it: all_trip_data.to_parquet('outputs/all_trip_data.parquet')
3. Run this script: python generate_per_line_graphs.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
DOCS_DIR = SCRIPT_DIR / "docs"
LINES_DIR = DOCS_DIR / "lines"
DATA_DIR = DOCS_DIR / "data"
DOCS_DIR.mkdir(exist_ok=True)
LINES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Tübingen bounding box
TUEBINGEN_LAT_MIN, TUEBINGEN_LAT_MAX = 48.47, 48.55
TUEBINGEN_LON_MIN, TUEBINGEN_LON_MAX = 9.02, 9.10

# Color settings
PUNCTUAL_THRESHOLD = 0.1
NEUTRAL_COLOR = "#888888"
cmap_delayed = plt.cm.YlOrRd


def load_trip_data():
    """Load trip data from parquet file."""
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        print("Please run the notebook and export all_trip_data:")
        print("  all_trip_data.to_parquet('outputs/all_trip_data.parquet')")
        return None
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} trip records")
    print(f"Columns: {df.columns.tolist()}")
    return df


def build_network_graph(trip_df: pd.DataFrame, line_filter: str = None):
    """Build network graph from trip data, optionally filtered by line."""
    
    df = trip_df.copy()
    
    # Filter out rows without valid coordinates FIRST
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    
    # Use departure_delay_minutes as delay_minutes
    if 'delay_minutes' not in df.columns:
        if 'departure_delay_minutes' in df.columns:
            df['delay_minutes'] = df['departure_delay_minutes']
        elif 'arrival_delay_minutes' in df.columns:
            df['delay_minutes'] = df['arrival_delay_minutes']
        else:
            df['delay_minutes'] = 0
    
    # Filter by line if specified
    if line_filter:
        df = df[df['line_name'] == line_filter]
    
    if df.empty or len(df) < 2:
        return None
    
    # Create edges from consecutive stops in each journey
    df = df.sort_values(['journey_ref', 'timestamp'])
    df['next_stop'] = df.groupby('journey_ref')['stop_name'].shift(-1)
    df['next_lat'] = df.groupby('journey_ref')['latitude'].shift(-1)
    df['next_lon'] = df.groupby('journey_ref')['longitude'].shift(-1)
    
    # Filter valid edges - must have valid coordinates for both stops
    edges = df.dropna(subset=['next_stop', 'next_lat', 'next_lon']).copy()
    edges = edges[edges['stop_name'] != edges['next_stop']]
    
    # Aggregate edges
    edge_agg = edges.groupby(['stop_name', 'next_stop']).agg(
        mean_delay=('delay_minutes', 'mean'),
        num_trips=('delay_minutes', 'count'),
        from_lat=('latitude', 'first'),
        from_lon=('longitude', 'first'),
        to_lat=('next_lat', 'first'),
        to_lon=('next_lon', 'first'),
    ).reset_index()
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes from unique stops
    stops = pd.concat([
        df[['stop_name', 'latitude', 'longitude', 'delay_minutes']].rename(
            columns={'stop_name': 'name', 'latitude': 'lat', 'longitude': 'lon', 'delay_minutes': 'delay'}
        )
    ]).groupby('name').agg(
        lat=('lat', 'mean'),
        lon=('lon', 'mean'),
        avg_delay=('delay', 'mean'),
    ).reset_index()
    
    for _, row in stops.iterrows():
        G.add_node(
            row['name'],
            label=row['name'],
            latitude=float(row['lat']),
            longitude=float(row['lon']),
            avg_delay=float(row['avg_delay']) if pd.notna(row['avg_delay']) else 0.0,
        )
    
    # Add edges
    for _, row in edge_agg.iterrows():
        if row['stop_name'] in G.nodes and row['next_stop'] in G.nodes:
            G.add_edge(
                row['stop_name'],
                row['next_stop'],
                mean_delay=float(row['mean_delay']) if pd.notna(row['mean_delay']) else 0.0,
                num_trips=int(row['num_trips']),
            )
    
    return G


def create_plotly_graph(G: nx.Graph, title: str, output_path: Path):
    """Create Plotly visualization for a graph."""
    
    if G is None or G.number_of_nodes() == 0:
        return
    
    pos = {node: (data["longitude"], data["latitude"]) for node, data in G.nodes(data=True)}
    
    # Get delays for normalization
    all_delays = sorted([data.get("mean_delay", 0) for u, v, data in G.edges(data=True)])
    if not all_delays:
        all_delays = [0]
    delayed_values = sorted([d for d in all_delays if d > PUNCTUAL_THRESHOLD])
    
    def delay_to_color(delay):
        if delay <= PUNCTUAL_THRESHOLD:
            return NEUTRAL_COLOR
        if delayed_values:
            rank = sum(1 for d in delayed_values if d < delay)
            delay_norm = rank / len(delayed_values)
        else:
            delay_norm = 0.5
        rgba = cmap_delayed(delay_norm)
        return mcolors.rgb2hex(rgba[:3])
    
    # Create edge traces
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        delay = data.get("mean_delay", 0)
        color = delay_to_color(delay)
        
        edge_traces.append(
            go.Scattermapbox(
                lon=[x0, x1],
                lat=[y0, y1],
                mode="lines",
                line=dict(width=2, color=color),
                hoverinfo="text",
                text=f"{u} → {v}<br>Delay: {delay:.1f} min",
                showlegend=False,
            )
        )
    
    # Node trace
    node_lon = [pos[node][0] for node in G.nodes()]
    node_lat = [pos[node][1] for node in G.nodes()]
    node_text = [f"{node}" for node in G.nodes()]
    
    node_trace = go.Scattermapbox(
        lon=node_lon,
        lat=node_lat,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(size=8, color="#1f77b4"),
        showlegend=False,
    )
    
    # Legend traces
    legend_traces = [
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=NEUTRAL_COLOR),
            name="Punctual (≤0.1 min)", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(0.0)[:3])),
            name="Slight delay", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(0.5)[:3])),
            name="Moderate delay", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(1.0)[:3])),
            name="Significant delay", showlegend=True,
        ),
    ]
    
    # Calculate center
    all_lats = [pos[n][1] for n in G.nodes()]
    all_lons = [pos[n][0] for n in G.nodes()]
    center_lat = np.mean(all_lats) if all_lats else 48.52
    center_lon = np.mean(all_lons) if all_lons else 9.05
    
    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=18)),
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)", font=dict(size=11),
            ),
            hovermode="closest",
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        ),
    )
    
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Saved: {output_path.name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def generate_index_html(lines: list):
    """Generate index.html with dropdown selector."""
    
    options_html = '<option value="all" selected>All Lines</option>\n'
    for line in sorted(lines):
        safe_name = line.replace("/", "_").replace(" ", "_")
        options_html += f'        <option value="{safe_name}">Line {line}</option>\n'
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tübingen Bus Network - Delay Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Georgia', 'Times New Roman', serif; 
            background: #ffffff; 
            padding: 1.5rem;
        }}
        .header {{
            background: #ffffff; 
            color: #333; 
            padding: 1rem 0 1.5rem 0;
            display: flex; 
            align-items: center; 
            justify-content: space-between;
            border-bottom: 1px solid #ddd;
            margin-bottom: 1rem;
        }}
        .header h1 {{ 
            font-size: 1.4rem; 
            font-weight: 400; 
            letter-spacing: 0.02em;
        }}
        .controls {{ display: flex; align-items: center; gap: 1rem; }}
        .controls label {{ font-size: 0.9rem; color: #555; }}
        .controls select {{
            padding: 0.4rem 0.8rem; 
            font-size: 0.9rem; 
            border: 1px solid #ccc;
            border-radius: 3px; 
            background: white; 
            cursor: pointer; 
            min-width: 140px;
            font-family: inherit;
        }}
        .controls select:focus {{ outline: 1px solid #666; }}
        .map-container {{ 
            width: 100%; 
            height: calc(100vh - 140px); 
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }}
        .map-container iframe {{ width: 100%; height: 100%; border: none; }}
        .info {{
            text-align: center;
            padding: 1rem 0 0 0;
            font-size: 0.8rem; 
            color: #888;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Tübingen Bus Network — Delay Analysis</h1>
        <div class="controls">
            <label for="line-select">Select Line:</label>
            <select id="line-select" onchange="loadLine(this.value)">
                {options_html}
            </select>
        </div>
    </div>
    <div class="map-container">
        <iframe id="map-frame" src="lines/network_all.html"></iframe>
    </div>
    <div class="info">Data: Nov 2025 – Jan 2026 | Source: TRIAS API</div>
    <script>
        function loadLine(value) {{
            document.getElementById('map-frame').src = 'lines/network_' + value + '.html';
        }}
    </script>
</body>
</html>
'''
    
    with open(DOCS_DIR / "index.html", "w") as f:
        f.write(html_content)
    print("  Saved: index.html")


def export_line_data(trip_df: pd.DataFrame, line_filter: str, output_path: Path):
    """Export edge data for a line as JSON for client-side table display."""
    
    df = trip_df.copy()
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    
    if 'delay_minutes' not in df.columns:
        if 'departure_delay_minutes' in df.columns:
            df['delay_minutes'] = df['departure_delay_minutes']
        elif 'arrival_delay_minutes' in df.columns:
            df['delay_minutes'] = df['arrival_delay_minutes']
        else:
            df['delay_minutes'] = 0
    
    if line_filter:
        df = df[df['line_name'] == line_filter]
    
    if df.empty:
        return
    
    # Create edges
    df = df.sort_values(['journey_ref', 'timestamp'])
    df['next_stop'] = df.groupby('journey_ref')['stop_name'].shift(-1)
    edges = df.dropna(subset=['next_stop']).copy()
    edges = edges[edges['stop_name'] != edges['next_stop']]
    
    # Aggregate
    edge_agg = edges.groupby(['stop_name', 'next_stop']).agg(
        mean_delay=('delay_minutes', 'mean'),
        num_trips=('delay_minutes', 'count'),
    ).reset_index()
    
    # Round values
    edge_agg['mean_delay'] = edge_agg['mean_delay'].round(2)
    
    # Convert to list of dicts
    data = {
        'edges': edge_agg.rename(columns={
            'stop_name': 'from',
            'next_stop': 'to',
            'mean_delay': 'delay_min',
            'num_trips': 'trips'
        }).to_dict(orient='records'),
        'summary': {
            'total_edges': len(edge_agg),
            'total_trips': int(edge_agg['num_trips'].sum()),
            'avg_delay': round(edge_agg['mean_delay'].mean(), 2) if len(edge_agg) > 0 else 0,
            'max_delay': round(edge_agg['mean_delay'].max(), 2) if len(edge_agg) > 0 else 0,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)


def main():
    print("=" * 70)
    print("Generating Per-Line Network Graphs for GitHub Pages")
    print("=" * 70)
    
    # Load data
    trip_df = load_trip_data()
    if trip_df is None:
        return
    
    # Get unique lines
    lines = sorted(trip_df['line_name'].dropna().unique())
    print(f"\nFound {len(lines)} lines: {', '.join(lines[:10])}{'...' if len(lines) > 10 else ''}")
    
    # Generate full network graph
    print("\nGenerating graphs and data...")
    G_full = build_network_graph(trip_df, line_filter=None)
    create_plotly_graph(G_full, "All Lines", LINES_DIR / "network_all.html")
    export_line_data(trip_df, None, DATA_DIR / "data_all.json")
    
    # Generate per-line graphs and data
    generated_lines = []
    for line in lines:
        G_line = build_network_graph(trip_df, line_filter=line)
        if G_line and G_line.number_of_nodes() > 1:
            safe_name = line.replace("/", "_").replace(" ", "_")
            create_plotly_graph(G_line, f"Line {line}", LINES_DIR / f"network_{safe_name}.html")
            export_line_data(trip_df, line, DATA_DIR / f"data_{safe_name}.json")
            generated_lines.append(line)
    
    # Generate index.html
    print("\nGenerating index.html...")
    generate_index_html(generated_lines)
    
    print("\n" + "=" * 70)
    print(f"Done! Generated {len(generated_lines) + 1} graphs + data files in docs/")
    print("To deploy: push docs/ to GitHub and enable GitHub Pages")
    print("=" * 70)


if __name__ == "__main__":
    main()
