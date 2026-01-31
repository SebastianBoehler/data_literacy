"""
Figure 3: Network Graph - Line-Specific Delay Variation

Shows Line 5 delay hotspots with edges colored by mean delay level.
Single panel visualization for the paper's line-specific variation section.

Uses map background like docs HTML graphs.

Outputs:
- plots/fig3_network_graph.html (interactive with map)
- plots/fig3_network_graph.png (static for paper)
- plots/fig3_network_graph.pdf (for paper)
- paper/images/fig3_network_graph.pdf
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from pathlib import Path
import sys

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent  # scripts/ -> code/
sys.path.insert(0, str(SCRIPT_DIR))

DATA_PATH = SCRIPT_DIR / "outputs" / "all_trip_data.parquet"
PLOT_DIR = SCRIPT_DIR / "plots"
PAPER_DIR = SCRIPT_DIR / "paper" / "images"

PLOT_DIR.mkdir(exist_ok=True)
PAPER_DIR.mkdir(exist_ok=True)

# Line to show - Line 4 has good direction balance (46%) and reasonable data quality
# Line 4: 16,038 records, 52% duplicate rate, H:10,965 R:5,073
FOCUS_LINE = '4'
SCHEDULE_CHANGE_DATE = pd.Timestamp("2025-12-14")

# Delay color thresholds (same as docs network graphs)
DELAY_BINS = [0, 1, 2, 5, 10, 20, float('inf')]  # minutes
NEUTRAL_COLOR = "#888888"
cmap_delayed = plt.cm.YlOrRd

# Tübingen bounding box for filtering
TUEBINGEN_LAT_MIN, TUEBINGEN_LAT_MAX = 48.49, 48.54
TUEBINGEN_LON_MIN, TUEBINGEN_LON_MAX = 9.03, 9.09


def load_data():
    """Load trip data from parquet file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} records")
    return df


def build_edges_for_period(data: pd.DataFrame):
    """Build edges from a single period's data (pre or post schedule change)."""
    
    if data.empty or len(data) < 2:
        return pd.DataFrame()
    
    # Sort by journey and stop sequence
    if 'stop_sequence' in data.columns:
        data = data.sort_values(['journey_ref', 'stop_sequence'])
    else:
        data = data.sort_values(['journey_ref', 'timestamp'])
    
    # Create edges from consecutive stops within each journey
    data = data.copy()
    data['next_stop'] = data.groupby('journey_ref')['stop_name'].shift(-1)
    data['next_lat'] = data.groupby('journey_ref')['latitude'].shift(-1)
    data['next_lon'] = data.groupby('journey_ref')['longitude'].shift(-1)
    
    # Filter valid edges
    edges = data.dropna(subset=['next_stop', 'next_lat', 'next_lon']).copy()
    edges = edges[edges['stop_name'] != edges['next_stop']]
    
    return edges


def build_network_graph(df: pd.DataFrame, line_filter: str = None, direction_filter: str = None):
    """Build network graph from trip data, optionally filtered by line and direction.
    
    Processes pre and post schedule change data separately to avoid spurious edges
    from route changes, then merges the edge statistics.
    
    Args:
        df: Trip data DataFrame
        line_filter: Filter to specific line (e.g., '5')
        direction_filter: Filter to specific direction ('H' for outbound, 'R' for return)
    """
    
    data = df.copy()
    
    # Filter out rows without valid coordinates
    data = data[data['latitude'].notna() & data['longitude'].notna()]
    
    # Ensure delay_minutes exists
    if 'delay_minutes' not in data.columns:
        if 'departure_delay_minutes' in data.columns:
            data['delay_minutes'] = data['departure_delay_minutes']
        elif 'arrival_delay_minutes' in data.columns:
            data['delay_minutes'] = data['arrival_delay_minutes']
        else:
            data['delay_minutes'] = 0
    
    # Filter by line if specified
    if line_filter:
        data = data[data['line_name'] == line_filter]
    
    # Extract and filter by direction if specified
    if direction_filter:
        data['direction'] = data['journey_ref'].str.extract(r'::([HR]):')[0].fillna('X')
        data = data[data['direction'] == direction_filter]
    
    if data.empty or len(data) < 2:
        return None
    
    # Split data by schedule change and build edges separately
    # This prevents spurious edges from route changes
    data_pre = data[data['timestamp'] < SCHEDULE_CHANGE_DATE]
    data_post = data[data['timestamp'] >= SCHEDULE_CHANGE_DATE]
    
    edges_pre = build_edges_for_period(data_pre)
    edges_post = build_edges_for_period(data_post)
    
    # Combine edges from both periods
    edges = pd.concat([edges_pre, edges_post], ignore_index=True)
    
    if edges.empty:
        return None
    
    # Aggregate edges - use mean delay
    edge_agg = edges.groupby(['stop_name', 'next_stop']).agg(
        mean_delay=('delay_minutes', 'mean'),
        from_lat=('latitude', 'first'),
        from_lon=('longitude', 'first'),
        to_lat=('next_lat', 'first'),
        to_lon=('next_lon', 'first'),
    ).reset_index()
    
    # Count actual rows
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
    
    # Filter out spurious edges caused by duplicate stop_sequence data:
    # - Single-trip edges (count=1): remove if distance > 1.0 km
    # - Low-count edges (count 2-3): remove if distance > 1.5 km  
    # - Higher-count edges (count >= 4): keep all (reliable data)
    spurious = (
        ((edge_agg['num_trips'] == 1) & (edge_agg['distance_km'] > 1.0)) |
        ((edge_agg['num_trips'] >= 2) & (edge_agg['num_trips'] <= 3) & (edge_agg['distance_km'] > 1.5))
    )
    edge_agg = edge_agg[~spurious]
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes
    stops = pd.concat([
        data[['stop_name', 'latitude', 'longitude', 'delay_minutes']].rename(
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


def delay_to_color(delay):
    """Map delay to color based on absolute thresholds (same as docs)."""
    DELAY_COLORS = [
        NEUTRAL_COLOR,  # ≤1 min (gray - punctual)
        mcolors.rgb2hex(cmap_delayed(0.1)[:3]),  # 1-2 min (light yellow)
        mcolors.rgb2hex(cmap_delayed(0.3)[:3]),  # 2-5 min (yellow)
        mcolors.rgb2hex(cmap_delayed(0.5)[:3]),  # 5-10 min (orange)
        mcolors.rgb2hex(cmap_delayed(0.75)[:3]), # 10-20 min (red-orange)
        mcolors.rgb2hex(cmap_delayed(1.0)[:3]),  # >20 min (dark red)
    ]
    for i, threshold in enumerate(DELAY_BINS[1:]):
        if delay < threshold:
            return DELAY_COLORS[i]
    return DELAY_COLORS[-1]


def create_single_line_map_traces(G, subplot_name="map"):
    """Create Plotly traces for a single line's network on a map."""
    
    if G is None or G.number_of_nodes() == 0:
        return []
    
    traces = []
    pos = {node: (data["longitude"], data["latitude"]) for node, data in G.nodes(data=True)}
    
    # Key stops to label
    KEY_STOPS = ['Hauptbahnhof', 'Neckarbrücke', 'Morgenstelle']
    
    # Add edges as lines colored by delay
    for u, v, data in G.edges(data=True):
        delay = data.get('mean_delay', 0)
        color = delay_to_color(delay)
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        traces.append(
            go.Scattermap(
                lon=[x0, x1], lat=[y0, y1],
                mode="lines",
                line=dict(width=3, color=color),  # Slightly thinner edges
                hoverinfo="text",
                text=f"{u} → {v}<br>Delay: {delay:.1f} min",
                showlegend=False,
                subplot=subplot_name,
            )
        )
    
    # Add nodes - separate key stops for labels
    key_nodes = [n for n in G.nodes() if any(k in n for k in KEY_STOPS)]
    other_nodes = [n for n in G.nodes() if n not in key_nodes]
    
    # Regular nodes (no labels)
    if other_nodes:
        node_lons = [pos[n][0] for n in other_nodes]
        node_lats = [pos[n][1] for n in other_nodes]
        node_text = [f"{n}<br>Avg delay: {G.nodes[n].get('avg_delay', 0):.1f} min" for n in other_nodes]
        
        traces.append(
            go.Scattermap(
                lon=node_lons, lat=node_lats,
                mode="markers",
                marker=dict(size=7, color="#1f77b4"),
                hoverinfo="text",
                text=node_text,
                showlegend=False,
                subplot=subplot_name,
            )
        )
    
    # Key nodes with labels
    if key_nodes:
        key_lons = [pos[n][0] for n in key_nodes]
        key_lats = [pos[n][1] for n in key_nodes]
        # Short labels and positions for key stops
        key_labels = []
        text_positions = []
        for n in key_nodes:
            if 'Hauptbahnhof' in n and 'Süd' not in n:
                key_labels.append('Hbf')
                text_positions.append('top center')
            elif 'Hauptbahnhof' in n and 'Süd' in n:
                key_labels.append('Hbf Süd')
                text_positions.append('bottom right')
            elif 'Neckarbrücke' in n:
                key_labels.append('Neckarbrücke')
                text_positions.append('top right')
            elif 'Morgenstelle' in n:
                key_labels.append('Morgenstelle')
                text_positions.append('top center')
            else:
                key_labels.append(n.replace('Tübingen ', ''))
                text_positions.append('top right')
        key_text = [f"{n}<br>Avg delay: {G.nodes[n].get('avg_delay', 0):.1f} min" for n in key_nodes]
        
        # Add each key node separately to control text position
        for i, n in enumerate(key_nodes):
            traces.append(
                go.Scattermap(
                    lon=[key_lons[i]], lat=[key_lats[i]],
                    mode="markers+text",
                    marker=dict(size=8, color="#1f77b4"),  # Same color as other nodes
                    text=[key_labels[i]],
                    textposition=text_positions[i],
                    textfont=dict(size=10, color="black"),
                    hoverinfo="text",
                    hovertext=[key_text[i]],
                    showlegend=False,
                    subplot=subplot_name,
                )
            )
    
    return traces


def create_direction_comparison_figure(G_h, G_r, stats_h, stats_r):
    """Create 2-panel figure comparing directions (H=outbound, R=return)."""
    
    # Get positions from both graphs
    pos_h = {node: (data["longitude"], data["latitude"]) 
             for node, data in G_h.nodes(data=True)} if G_h else {}
    pos_r = {node: (data["longitude"], data["latitude"]) 
             for node, data in G_r.nodes(data=True)} if G_r else {}
    
    # Calculate center from combined positions
    all_pos = {**pos_h, **pos_r}
    if all_pos:
        center_lat = np.mean([p[1] for p in all_pos.values()])
        center_lon = np.mean([p[0] for p in all_pos.values()])
    else:
        center_lat, center_lon = 48.52, 9.05
    
    # Create subplots with map
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "(A) Direction H (Outbound)",
            "(B) Direction R (Return)"
        ],
        specs=[[{"type": "scattermap"}, {"type": "scattermap"}]],
        horizontal_spacing=0.02,
    )
    
    # Add traces for direction H (left panel) - uses "map"
    h_traces = create_single_line_map_traces(G_h, subplot_name="map")
    for trace in h_traces:
        fig.add_trace(trace)
    
    # Add traces for direction R (right panel) - uses "map2"
    r_traces = create_single_line_map_traces(G_r, subplot_name="map2")
    for trace in r_traces:
        fig.add_trace(trace)
    
    # Add legend traces (only once) - assign to first map
    DELAY_COLORS = [
        NEUTRAL_COLOR,
        mcolors.rgb2hex(cmap_delayed(0.1)[:3]),
        mcolors.rgb2hex(cmap_delayed(0.3)[:3]),
        mcolors.rgb2hex(cmap_delayed(0.5)[:3]),
        mcolors.rgb2hex(cmap_delayed(0.75)[:3]),
        mcolors.rgb2hex(cmap_delayed(1.0)[:3]),
    ]
    DELAY_LABELS = ['≤1 min', '1-2 min', '2-5 min', '5-10 min', '10-20 min', '>20 min']
    
    for color, label in zip(DELAY_COLORS, DELAY_LABELS):
        fig.add_trace(
            go.Scattermap(
                lon=[None], lat=[None], mode="markers",
                marker=dict(size=12, color=color),
                name=label, showlegend=True,
                subplot="map",
            )
        )
    
    # Update layout with explicit map configurations for each subplot
    fig.update_layout(
        title=None,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.9)", font=dict(size=10),
            title="Mean Delay",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        width=1400,
        height=700,
        # Configure first map (left panel)
        map=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12.5,
            domain=dict(x=[0, 0.48], y=[0, 1]),
        ),
        # Configure second map (right panel)
        map2=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12.5,
            domain=dict(x=[0.52, 1], y=[0, 1]),
        ),
    )
    
    return fig


def main():
    print("=" * 60)
    print("FIGURE 3: Line-Specific Delay Variation by Direction")
    print("=" * 60)
    
    df = load_data()
    
    # Extract direction for statistics
    df['direction'] = df['journey_ref'].str.extract(r'::([HR]):')[0].fillna('X')
    
    # Build graphs for each direction
    print(f"\nBuilding graphs for Line {FOCUS_LINE} by direction...")
    
    G_h = build_network_graph(df, line_filter=FOCUS_LINE, direction_filter='H')
    G_r = build_network_graph(df, line_filter=FOCUS_LINE, direction_filter='R')
    
    if G_h:
        print(f"  Direction H: {G_h.number_of_nodes()} nodes, {G_h.number_of_edges()} edges")
    else:
        print("  Direction H: No data")
    
    if G_r:
        print(f"  Direction R: {G_r.number_of_nodes()} nodes, {G_r.number_of_edges()} edges")
    else:
        print("  Direction R: No data")
    
    if not G_h and not G_r:
        print("  ERROR: No data for either direction")
        return
    
    # Calculate statistics per direction
    line_h = df[(df['line_name'] == FOCUS_LINE) & (df['direction'] == 'H')]['delay_minutes'].dropna()
    line_r = df[(df['line_name'] == FOCUS_LINE) & (df['direction'] == 'R')]['delay_minutes'].dropna()
    
    stats_h = {'mean': line_h.mean(), 'median': line_h.median(), 'count': len(line_h)}
    stats_r = {'mean': line_r.mean(), 'median': line_r.median(), 'count': len(line_r)}
    
    print(f"\n  Direction H: mean={stats_h['mean']:.2f} min, n={stats_h['count']:,}")
    print(f"  Direction R: mean={stats_r['mean']:.2f} min, n={stats_r['count']:,}")
    
    # Create direction comparison figure
    print("\nCreating visualization...")
    fig = create_direction_comparison_figure(G_h, G_r, stats_h, stats_r)
    
    if fig is None:
        print("ERROR: Could not create figure")
        return
    
    # Save interactive HTML
    out_html = PLOT_DIR / "fig3_network_graph.html"
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    print(f"Saved: {out_html}")
    
    # Save static images (PNG and PDF)
    out_png = PLOT_DIR / "fig3_network_graph.png"
    out_pdf = PLOT_DIR / "fig3_network_graph.pdf"
    
    try:
        fig.write_image(str(out_png), scale=2)
        print(f"Saved: {out_png}")
        
        fig.write_image(str(out_pdf))
        print(f"Saved: {out_pdf}")
        
        # Also save to paper/images/ for LaTeX
        paper_pdf = PAPER_DIR / "fig3_network_graph.pdf"
        fig.write_image(str(paper_pdf))
        print(f"Saved: {paper_pdf}")
    except Exception as e:
        print(f"Note: Could not save static images ({e})")
        print("  Install kaleido for static export: pip install -U kaleido")
        print("  HTML version saved successfully - use browser screenshot if needed")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Line {FOCUS_LINE} delay by direction:")
    print(f"  Direction H (Outbound): mean={stats_h['mean']:.2f} min, n={stats_h['count']:,}")
    print(f"  Direction R (Return):   mean={stats_r['mean']:.2f} min, n={stats_r['count']:,}")


if __name__ == "__main__":
    main()
