"""
Generate per-line network graphs for GitHub Pages visualization.
Uses all_trip_data from the notebook (exported as parquet) for accurate results.

Generates artifacts for three time periods:
- all: All data
- pre: Before schedule change (Dec 14, 2025)
- post: After schedule change

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
SCRIPT_DIR = Path(__file__).parent.parent  # docs_generation/ -> code/
DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
DOCS_DIR = SCRIPT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# Schedule change date for period filtering
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")

# Period configurations
PERIODS = {
    "all": {"label": "All Data", "filter": None},
    "pre": {"label": "Before Schedule Change", "filter": "pre"},
    "post": {"label": "After Schedule Change", "filter": "post"},
}

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


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter dataframe by time period."""
    if period == "pre":
        return df[df['timestamp'] < SCHEDULE_CHANGE_DATE].copy()
    elif period == "post":
        return df[df['timestamp'] >= SCHEDULE_CHANGE_DATE].copy()
    return df.copy()  # "all" - no filter


def _build_edges_for_period(data: pd.DataFrame):
    """Build edges from a single period's data (pre or post schedule change).
    
    This helper prevents spurious edges when combining data from different schedule periods.
    """
    if data.empty or len(data) < 2:
        return pd.DataFrame()
    
    data = data.copy()
    
    # Use stop_sequence for ordering
    if 'stop_sequence' in data.columns:
        data = data.sort_values(['journey_ref', 'stop_sequence'])
    else:
        data = data.sort_values(['journey_ref', 'timestamp'])
    
    # Create edges from consecutive stops within each journey
    data['next_stop'] = data.groupby('journey_ref')['stop_name'].shift(-1)
    data['next_lat'] = data.groupby('journey_ref')['latitude'].shift(-1)
    data['next_lon'] = data.groupby('journey_ref')['longitude'].shift(-1)
    
    # Filter valid edges
    edges = data.dropna(subset=['next_stop', 'next_lat', 'next_lon']).copy()
    edges = edges[edges['stop_name'] != edges['next_stop']]
    
    return edges


def build_network_graph(trip_df: pd.DataFrame, line_filter: str = None):
    """Build network graph from trip data, optionally filtered by line.
    
    Processes pre and post schedule change data separately to avoid spurious edges
    from route changes, then merges the edge statistics.
    """
    
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
    
    # Extract direction from journey_ref (H=Hin/outbound, R=Rück/return) if available
    df['direction'] = df['journey_ref'].str.extract(r'::([HR]):')[0].fillna('X')
    
    # Split data by schedule change and build edges separately
    # This prevents spurious edges from route changes when combining all data
    df_pre = df[df['timestamp'] < SCHEDULE_CHANGE_DATE]
    df_post = df[df['timestamp'] >= SCHEDULE_CHANGE_DATE]
    
    edges_pre = _build_edges_for_period(df_pre)
    edges_post = _build_edges_for_period(df_post)
    
    # Combine edges from both periods
    edges = pd.concat([edges_pre, edges_post], ignore_index=True)
    
    if edges.empty:
        return None
    
    # Aggregate edges - use 'size' for count to include rows with NaN delays
    edge_agg = edges.groupby(['stop_name', 'next_stop']).agg(
        mean_delay=('delay_minutes', 'mean'),
        from_lat=('latitude', 'first'),
        from_lon=('longitude', 'first'),
        to_lat=('next_lat', 'first'),
        to_lon=('next_lon', 'first'),
    ).reset_index()
    # Count actual rows (not just non-NaN delay values)
    edge_counts = edges.groupby(['stop_name', 'next_stop']).size().reset_index(name='num_trips')
    edge_agg = edge_agg.merge(edge_counts, on=['stop_name', 'next_stop'])
    
    # Calculate edge distance to filter spurious long-distance edges
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    edge_agg['distance_km'] = haversine(
        edge_agg['from_lat'].values, edge_agg['from_lon'].values,
        edge_agg['to_lat'].values, edge_agg['to_lon'].values
    )
    
    # Filter out spurious edges caused by missing coordinates creating "jumps"
    # Rules based on trip count and distance:
    # - Single-trip edges (count=1): remove if distance > 1.0 km
    # - Low-count edges (count 2-3): remove if distance > 1.5 km  
    # - Higher-count edges (count >= 4): keep all (reliable data)
    # Note: We don't filter by delay since some lines genuinely have high delays
    spurious = (
        ((edge_agg['num_trips'] == 1) & (edge_agg['distance_km'] > 1.0)) |
        ((edge_agg['num_trips'] >= 2) & (edge_agg['num_trips'] <= 3) & (edge_agg['distance_km'] > 1.5))
    )
    edge_agg = edge_agg[~spurious]
    
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
    
    # Absolute delay thresholds for consistent coloring across all lines
    # Delays are mapped to fixed ranges for interpretability
    DELAY_BINS = [0, 1, 2, 5, 10, 20, float('inf')]  # minutes
    DELAY_LABELS = ['≤1 min', '1-2 min', '2-5 min', '5-10 min', '10-20 min', '>20 min']
    DELAY_COLORS = [
        NEUTRAL_COLOR,  # ≤1 min (gray - punctual)
        mcolors.rgb2hex(cmap_delayed(0.1)[:3]),  # 1-2 min (light yellow)
        mcolors.rgb2hex(cmap_delayed(0.3)[:3]),  # 2-5 min (yellow)
        mcolors.rgb2hex(cmap_delayed(0.5)[:3]),  # 5-10 min (orange)
        mcolors.rgb2hex(cmap_delayed(0.75)[:3]), # 10-20 min (red-orange)
        mcolors.rgb2hex(cmap_delayed(1.0)[:3]),  # >20 min (dark red)
    ]
    
    def delay_to_color(delay):
        """Map delay to color based on absolute thresholds."""
        for i, threshold in enumerate(DELAY_BINS[1:]):
            if delay < threshold:
                return DELAY_COLORS[i]
        return DELAY_COLORS[-1]
    
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
    
    # Legend traces with absolute delay ranges
    legend_traces = [
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[0]),
            name="≤1 min (punctual)", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[1]),
            name="1-2 min", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[2]),
            name="2-5 min", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[3]),
            name="5-10 min", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[4]),
            name="10-20 min", showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=DELAY_COLORS[5]),
            name=">20 min", showlegend=True,
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
    
    # Create edges - use stop_sequence for proper ordering
    if 'stop_sequence' in df.columns:
        df = df.sort_values(['journey_ref', 'stop_sequence'])
    else:
        df = df.sort_values(['journey_ref', 'timestamp'])
    df['next_stop'] = df.groupby('journey_ref')['stop_name'].shift(-1)
    df['next_lat'] = df.groupby('journey_ref')['latitude'].shift(-1)
    df['next_lon'] = df.groupby('journey_ref')['longitude'].shift(-1)
    edges = df.dropna(subset=['next_stop', 'next_lat', 'next_lon']).copy()
    edges = edges[edges['stop_name'] != edges['next_stop']]
    
    # Aggregate - use 'size' for count to include rows with NaN delays
    edge_agg = edges.groupby(['stop_name', 'next_stop']).agg(
        mean_delay=('delay_minutes', 'mean'),
        from_lat=('latitude', 'first'),
        from_lon=('longitude', 'first'),
        to_lat=('next_lat', 'first'),
        to_lon=('next_lon', 'first'),
    ).reset_index()
    # Count actual rows (not just non-NaN delay values)
    edge_counts = edges.groupby(['stop_name', 'next_stop']).size().reset_index(name='num_trips')
    edge_agg = edge_agg.merge(edge_counts, on=['stop_name', 'next_stop'])
    
    # Calculate distance for filtering
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    edge_agg['distance_km'] = haversine(
        edge_agg['from_lat'].values, edge_agg['from_lon'].values,
        edge_agg['to_lat'].values, edge_agg['to_lon'].values
    )
    
    # Filter spurious edges (same rules as network graph)
    spurious = (
        ((edge_agg['num_trips'] == 1) & (edge_agg['distance_km'] > 1.0)) |
        ((edge_agg['num_trips'] >= 2) & (edge_agg['num_trips'] <= 3) & (edge_agg['distance_km'] > 1.5))
    )
    edge_agg = edge_agg[~spurious]
    
    # Filter out edges with 0 trips (no actual data)
    edge_agg = edge_agg[edge_agg['num_trips'] > 0].copy()
    
    # Round values
    edge_agg['mean_delay'] = edge_agg['mean_delay'].round(2)
    
    # Convert to list of dicts
    edges_list = edge_agg.rename(columns={
        'stop_name': 'from',
        'next_stop': 'to',
        'mean_delay': 'delay_min',
        'num_trips': 'trips'
    }).to_dict(orient='records')

    # Ensure all values are JSON-serializable (NaN -> None for valid JSON)
    for edge in edges_list:
        edge['trips'] = int(edge['trips'])
        if pd.isna(edge.get('delay_min')):
            edge['delay_min'] = None
    
    avg_delay = edge_agg['mean_delay'].mean()
    max_delay = edge_agg['mean_delay'].max()
    
    data = {
        'edges': edges_list,
        'summary': {
            'total_edges': len(edge_agg),
            'total_trips': int(edge_agg['num_trips'].sum()),
            'avg_delay': round(avg_delay, 2) if pd.notna(avg_delay) else 0,
            'max_delay': round(max_delay, 2) if pd.notna(max_delay) else 0,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)


def generate_period_artifacts(trip_df: pd.DataFrame, period: str, period_label: str):
    """Generate all artifacts for a specific time period."""
    
    # Create period-specific directories
    lines_dir = DOCS_DIR / "lines" / period
    data_dir = DOCS_DIR / "data" / period
    lines_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data by period
    df = filter_by_period(trip_df, period)
    print(f"\n  {period_label}: {len(df):,} records")
    
    if df.empty:
        print(f"  WARNING: No data for period '{period}'")
        return []
    
    # Get unique lines
    lines = sorted(df['line_name'].dropna().unique())
    
    # Generate full network graph
    G_full = build_network_graph(df, line_filter=None)
    create_plotly_graph(G_full, f"All Lines ({period_label})", lines_dir / "network_all.html")
    export_line_data(df, None, data_dir / "data_all.json")
    
    # Generate per-line graphs and data
    generated_lines = []
    for line in lines:
        G_line = build_network_graph(df, line_filter=line)
        if G_line and G_line.number_of_nodes() > 1:
            safe_name = line.replace("/", "_").replace(" ", "_")
            create_plotly_graph(G_line, f"Line {line} ({period_label})", lines_dir / f"network_{safe_name}.html")
            export_line_data(df, line, data_dir / f"data_{safe_name}.json")
            generated_lines.append(line)
    
    return generated_lines


def main():
    print("=" * 70)
    print("Generating Per-Line Network Graphs for GitHub Pages")
    print("(with period filtering: all, pre, post)")
    print("=" * 70)
    
    # Load data
    trip_df = load_trip_data()
    if trip_df is None:
        return
    
    # Get unique lines (for index.html)
    all_lines = sorted(trip_df['line_name'].dropna().unique())
    print(f"\nFound {len(all_lines)} lines: {', '.join(all_lines[:10])}{'...' if len(all_lines) > 10 else ''}")
    
    # Generate artifacts for each period
    print("\nGenerating graphs and data for each period...")
    for period, config in PERIODS.items():
        print(f"\n{'='*50}")
        print(f"Period: {period} ({config['label']})")
        print(f"{'='*50}")
        generate_period_artifacts(trip_df, period, config['label'])
    
    # Note: index.html is now maintained as a standalone file in docs/
    # (not generated from Python to reduce code duplication)
    print("\nNote: index.html is maintained as a standalone file in docs/")
    
    print("\n" + "=" * 70)
    print(f"Done! Generated artifacts for {len(PERIODS)} periods in docs/")
    print("Structure: docs/lines/{all,pre,post}/, docs/data/{all,pre,post}/")
    print("To deploy: push docs/ to GitHub and enable GitHub Pages")
    print("=" * 70)


if __name__ == "__main__":
    main()
