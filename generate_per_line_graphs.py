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
            background: #fafafa;
            color: #333;
            line-height: 1.7;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 3rem 2rem;
            background: #fff;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            padding-bottom: 2rem;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 2rem;
        }}
        .header h1 {{ 
            font-size: 1.8rem; 
            font-weight: 400; 
            letter-spacing: 0.02em;
            margin-bottom: 0.5rem;
        }}
        .header .subtitle {{
            font-size: 1rem;
            color: #666;
            font-style: italic;
        }}
        .section {{
            margin-bottom: 2.5rem;
        }}
        .section h2 {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #222;
        }}
        .section p {{
            font-size: 0.95rem;
            color: #444;
            margin-bottom: 1rem;
        }}
        .controls {{ 
            display: flex; 
            align-items: center; 
            gap: 1rem; 
            margin-bottom: 1rem;
        }}
        .controls label {{ font-size: 0.9rem; color: #555; }}
        .controls select {{
            padding: 0.4rem 0.8rem; 
            font-size: 0.9rem; 
            border: 1px solid #ccc;
            border-radius: 3px; 
            background: white; 
            cursor: pointer; 
            min-width: 160px;
            font-family: inherit;
        }}
        .controls select:focus {{ outline: 1px solid #666; }}
        .map-container {{ 
            width: 100%; 
            height: 500px;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .map-container iframe {{ width: 100%; height: 100%; border: none; }}
        .caption {{
            font-size: 0.85rem;
            color: #666;
            font-style: italic;
            text-align: center;
        }}
        .data-summary {{
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background: #f8f8f8;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        .data-table-container {{
            margin-top: 1rem;
        }}
        .data-table-container summary {{
            cursor: pointer;
            font-size: 0.9rem;
            color: #555;
            padding: 0.5rem 0;
        }}
        .table-wrapper {{
            max-height: 300px;
            overflow-y: auto;
            margin-top: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        #edge-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        #edge-table th, #edge-table td {{
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        #edge-table th {{
            background: #f5f5f5;
            position: sticky;
            top: 0;
            font-weight: 600;
        }}
        #edge-table tr:hover {{
            background: #fafafa;
        }}
        .footer {{
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid #e0e0e0;
            margin-top: 2rem;
            font-size: 0.8rem; 
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Tübingen Bus Network</h1>
            <p class="subtitle">Delay Analysis — Data Literacy Project</p>
        </div>

        <div class="section">
            <h2>Overview</h2>
            <p>
                This project analyzes public transit delays in the Tübingen bus network using real-time data 
                collected from the TRIAS API between November 2025 and January 2026. The visualization below 
                shows the network topology with edges colored by average delay: gray indicates punctual service, 
                while yellow to red indicates increasing delays.
            </p>
        </div>

        <div class="section">
            <h2>Interactive Network Map</h2>
            <div class="controls">
                <label for="line-select">Select Line:</label>
                <select id="line-select" onchange="loadLine(this.value)">
                    {options_html}
                </select>
            </div>
            <div class="map-container">
                <iframe id="map-frame" src="lines/network_all.html"></iframe>
            </div>
            <p class="caption">Figure 1: Interactive network map showing bus routes and average delays per segment.</p>
            
            <div class="data-summary" id="data-summary">
                <p><strong>Summary:</strong> <span id="summary-text">Loading...</span></p>
            </div>
            
            <details class="data-table-container">
                <summary>View Edge Data Table</summary>
                <div class="table-wrapper">
                    <table id="edge-table">
                        <thead>
                            <tr>
                                <th>From</th>
                                <th>To</th>
                                <th>Avg Delay (min)</th>
                                <th>Trips</th>
                            </tr>
                        </thead>
                        <tbody id="edge-table-body">
                        </tbody>
                    </table>
                </div>
            </details>
        </div>

        <div class="section">
            <h2>Methodology</h2>
            <p>
                Data was collected continuously from the TRIAS real-time transit API, capturing planned and 
                estimated arrival/departure times for all bus stops in the Tübingen network. Delays are 
                calculated as the difference between estimated and planned times. The network graph is 
                constructed by connecting consecutive stops within each journey, with edge weights representing 
                the mean delay observed on that segment.
            </p>
        </div>

        <div class="section">
            <h2>Key Findings</h2>
            <p>
                The analysis reveals that most bus segments operate with minimal delays (under 1 minute on average), 
                indicated by gray coloring in the network map. However, certain routes and time periods show 
                consistently higher delays, particularly during peak hours and on routes passing through the 
                city center. The interactive visualization allows exploration of delay patterns by individual 
                bus line.
            </p>
        </div>

        <div class="footer">
            Data: November 2025 – January 2026 | Source: TRIAS API | University of Tübingen
        </div>
    </div>

    <script>
        function loadLine(value) {{
            document.getElementById('map-frame').src = 'lines/network_' + value + '.html';
            loadData(value);
        }}
        
        function loadData(value) {{
            const dataUrl = 'data/data_' + value + '.json';
            fetch(dataUrl)
                .then(response => response.json())
                .then(data => {{
                    const summary = data.summary;
                    document.getElementById('summary-text').textContent = 
                        `${{summary.total_edges}} segments, ${{summary.total_trips.toLocaleString()}} trips observed, ` +
                        `avg delay: ${{summary.avg_delay}} min, max delay: ${{summary.max_delay}} min`;
                    
                    const tbody = document.getElementById('edge-table-body');
                    tbody.innerHTML = '';
                    data.edges.forEach(edge => {{
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${{edge.from}}</td>
                            <td>${{edge.to}}</td>
                            <td>${{edge.delay_min}}</td>
                            <td>${{edge.trips.toLocaleString()}}</td>
                        `;
                        tbody.appendChild(row);
                    }});
                }})
                .catch(err => {{
                    document.getElementById('summary-text').textContent = 'Data not available';
                    document.getElementById('edge-table-body').innerHTML = '';
                }});
        }}
        
        loadData('all');
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
