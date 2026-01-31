"""
Geographic Network Graph Visualization for Tübingen Bus Network

Creates a network graph using real geographic coordinates (latitude/longitude)
discovered via TRIAS API and trip data from Google Cloud Storage.

Output: Interactive HTML visualization using ipysigma/plotly
"""

import os
import io
import re
import pandas as pd
import networkx as nx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import TRIAS client for coordinate discovery
from trias import TriasClient

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
OUTPUT_HTML = SCRIPT_DIR / "geo_network_graph.html"

# GCS Configuration
GCS_KEY_FILE = SCRIPT_DIR / "departure-data-reader-key.json"
BUCKET_NAME = "departure_data"

# TRIAS Configuration
TRIAS_REQUESTOR_REF = "SeBaSTiaN_BoeHLeR"

# Tübingen area bounding box (approximate)
TUEBINGEN_LAT_MIN = 48.47
TUEBINGEN_LAT_MAX = 48.55
TUEBINGEN_LON_MIN = 9.02
TUEBINGEN_LON_MAX = 9.10

# Discovery center points (edge/border coordinates for broader coverage)
# These are spread across the Tübingen area to discover more stops
DISCOVERY_CENTERS = [
    # Main Tübingen center
    (48.5162, 9.0539),   # Hauptbahnhof
    (48.5225, 9.0585),   # Wilhelmstraße
    (48.5243, 9.0601),   # Uni/Neue Aula
    # North edge
    (48.5392, 9.0489),   # Ferd.-Chr.-Baur-Str.
    (48.5416, 9.0469),   # Kunsthalle
    (48.5382, 9.0566),   # Max-Planck-Institute
    # South edge  
    (48.4999, 9.0501),   # Danziger Straße
    (48.5037, 9.0499),   # Derendingen Bhf
    (48.4955, 9.0579),   # Ernst-Simon-Straße
    # East edge
    (48.5198, 9.0785),   # Neckarsulmer Str.
    (48.5218, 9.0821),   # Düsseldorfer Str.
    (48.5258, 9.0857),   # Aeulehöfe
    # West edge
    (48.5204, 9.0290),   # Weststadt
    (48.5204, 9.0401),   # Westbahnhof
    (48.5239, 9.0395),   # Herrenberger Str.
    # Weilheim/Kreßbach area (south)
    (48.4800, 9.0550),   # Kreßbach area
    (48.4850, 9.0500),   # Weilheim area
]


def init_gcs_client():
    """Initialize Google Cloud Storage client."""
    from google.cloud import storage
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(GCS_KEY_FILE)
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    print("Google Cloud Storage client initialized successfully.")
    return bucket


def list_trip_files(bucket) -> list:
    """List all trip_calls CSV files in the bucket."""
    blobs = list(bucket.list_blobs())
    trip_files = [blob.name for blob in blobs if 'trip_calls' in blob.name and blob.name.endswith('.csv')]
    print(f"Found {len(trip_files)} trip files")
    return trip_files


def download_and_combine_data(bucket, file_list: list, max_workers: int = 20, max_files: int = None) -> pd.DataFrame:
    """
    Download CSV files from GCS in parallel and combine into a single DataFrame.
    
    Args:
        bucket: GCS bucket object
        file_list: List of file names to download
        max_workers: Maximum parallel workers
        max_files: Optional limit on number of files to process (for faster testing)
    """
    if max_files:
        file_list = file_list[:max_files]
    
    total_files = len(file_list)
    if total_files == 0:
        print("No files to process.")
        return pd.DataFrame()

    print(f"Processing {total_files} files in parallel (max_workers={max_workers})...")

    def _load_single_file(file_name):
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data))

        # Extract timestamp from filename
        match = re.search(r'(\d{8}_\d{6})', file_name)
        if match:
            timestamp_str = match.group(1)
            df['timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M%S')
        else:
            df['timestamp'] = pd.NaT

        return df

    dfs = []
    workers = min(max_workers, total_files)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(_load_single_file, file_name): file_name
            for file_name in file_list
        }

        for i, future in enumerate(as_completed(future_to_file), 1):
            file_name = future_to_file[future]
            try:
                df = future.result()
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
            if i % 200 == 0 or i == total_files:
                print(f"Processed {i}/{total_files} files...")

    if not dfs:
        print("No dataframes were loaded successfully.")
        return pd.DataFrame()

    combined_data = pd.concat(dfs, ignore_index=True)

    print(f"\nCombined {total_files} files into DataFrame with {len(combined_data)} rows.")
    if 'timestamp' in combined_data.columns:
        print(f"Date range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")

    return combined_data


def discover_stop_coordinates() -> pd.DataFrame:
    """
    Discover stop coordinates using TRIAS API from multiple center points.
    Uses edge/border coordinates for broader coverage across Tübingen area.
    """
    print(f"\nDiscovering stop coordinates via TRIAS API...")
    print(f"  Using {len(DISCOVERY_CENTERS)} center points for discovery")
    
    trias_client = TriasClient(requestor_ref=TRIAS_REQUESTOR_REF)
    
    discovered_stops = {}  # stop_id -> {stop_id, stop_name, latitude, longitude}
    
    for i, (lat, lon) in enumerate(DISCOVERY_CENTERS, 1):
        try:
            stops = trias_client.fetch_stops(center=(lat, lon), radius_km=3, max_results=200)
            new_count = 0
            for _, s in stops.iterrows():
                stop_id = s.get('stop_id')
                if stop_id and stop_id not in discovered_stops:
                    if pd.notna(s.get('latitude')) and pd.notna(s.get('longitude')):
                        discovered_stops[stop_id] = {
                            'stop_id': stop_id,
                            'stop_name': s.get('stop_name'),
                            'latitude': s.get('latitude'),
                            'longitude': s.get('longitude'),
                        }
                        new_count += 1
            print(f"  Center {i}/{len(DISCOVERY_CENTERS)} ({lat:.4f}, {lon:.4f}): +{new_count} new stops")
        except Exception as e:
            print(f"  Center {i}/{len(DISCOVERY_CENTERS)} ({lat:.4f}, {lon:.4f}): Error - {e}")
    
    coords_df = pd.DataFrame(discovered_stops.values())
    print(f"\n  Total discovered stops with coordinates: {len(coords_df)}")
    
    return coords_df


def process_trip_data(all_trip_data: pd.DataFrame, coords_df: pd.DataFrame) -> tuple:
    """
    Process trip data to extract stop metadata and edge aggregations.
    
    Args:
        all_trip_data: Raw trip data from GCS
        coords_df: Stop coordinates discovered via TRIAS API
    
    Returns:
        tuple: (stop_meta DataFrame, edge_agg DataFrame)
    """
    print(f"\nRaw trip data: {len(all_trip_data):,} rows")

    # =============================================================================
    # STEP 1: Deduplicate trip data
    # =============================================================================
    trip_deduped = all_trip_data.copy()

    # Create unique trip identifier
    trip_deduped['trip_id'] = (
        trip_deduped['journey_ref'] + '_' + trip_deduped['operating_day_ref'].astype(str)
    )

    # Sort by timestamp descending to keep latest observation
    trip_deduped = trip_deduped.sort_values('timestamp', ascending=False)

    # Drop duplicates, keeping first (latest) per (trip, stop)
    trip_deduped = trip_deduped.drop_duplicates(
        subset=['trip_id', 'stop_point_ref', 'stop_sequence'],
        keep='first'
    )

    print(f"After deduplication: {len(trip_deduped):,} rows "
          f"({100*(1-len(trip_deduped)/len(all_trip_data)):.1f}% reduction)")

    # =============================================================================
    # STEP 2: Join coordinates from discovered_stops file
    # =============================================================================
    # Extract base stop_id from stop_point_ref (remove platform suffix like :0:1)
    trip_deduped['stop_id_base'] = trip_deduped['stop_point_ref'].str.extract(r'^(de:\d+:\d+)')[0]
    
    # Join coordinates by stop_id
    trip_deduped = trip_deduped.merge(
        coords_df[['stop_id', 'latitude', 'longitude']],
        left_on='stop_id_base',
        right_on='stop_id',
        how='left'
    )
    
    # Also try joining by stop_name for any remaining missing coords
    coords_by_name = coords_df.groupby('stop_name', as_index=False).agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).rename(columns={'latitude': 'lat_name', 'longitude': 'lon_name'})
    
    trip_deduped = trip_deduped.merge(coords_by_name, on='stop_name', how='left')
    
    # Fill missing coordinates from name-based lookup
    trip_deduped['latitude'] = trip_deduped['latitude'].fillna(trip_deduped['lat_name'])
    trip_deduped['longitude'] = trip_deduped['longitude'].fillna(trip_deduped['lon_name'])
    
    # Clean up temp columns
    trip_deduped = trip_deduped.drop(columns=['stop_id_base', 'stop_id', 'lat_name', 'lon_name'], errors='ignore')
    
    # Filter to stops with coordinates
    trip_filtered = trip_deduped[
        trip_deduped['latitude'].notna() & trip_deduped['longitude'].notna()
    ].copy()

    print(f"After joining coordinates: {len(trip_filtered):,} rows with coords")

    # =============================================================================
    # STEP 3: Filter to Tübingen area
    # =============================================================================
    tuebingen_mask = (
        (trip_filtered['latitude'] >= TUEBINGEN_LAT_MIN) &
        (trip_filtered['latitude'] <= TUEBINGEN_LAT_MAX) &
        (trip_filtered['longitude'] >= TUEBINGEN_LON_MIN) &
        (trip_filtered['longitude'] <= TUEBINGEN_LON_MAX)
    ) | trip_filtered['stop_name'].str.startswith('Tübingen', na=False)

    trip_filtered = trip_filtered[tuebingen_mask].copy()

    print(f"After filtering to Tübingen area: {len(trip_filtered):,} rows")
    print(f"Unique stops (by stop_point_ref): {trip_filtered['stop_point_ref'].nunique()}")

    # =============================================================================
    # STEP 4: Create unified stop key (by stop_name + rounded coordinates)
    # =============================================================================
    trip_filtered['lat_rounded'] = trip_filtered['latitude'].round(4)
    trip_filtered['lon_rounded'] = trip_filtered['longitude'].round(4)

    trip_filtered['unified_stop'] = (
        trip_filtered['stop_name'] + '_' +
        trip_filtered['lat_rounded'].astype(str) + '_' +
        trip_filtered['lon_rounded'].astype(str)
    )

    print(f"Unique stops after unification (by name+coords): {trip_filtered['unified_stop'].nunique()}")

    # =============================================================================
    # STEP 5: Prepare data for graph construction
    # =============================================================================
    trip_filtered['stop_sequence'] = pd.to_numeric(
        trip_filtered['stop_sequence'], errors='coerce'
    )
    trip_filtered = trip_filtered.dropna(
        subset=['journey_ref', 'stop_point_ref', 'stop_sequence']
    )

    for col in ['departure_delay_minutes', 'arrival_delay_minutes']:
        if col in trip_filtered.columns:
            trip_filtered[col] = pd.to_numeric(trip_filtered[col], errors='coerce')

    # Prefer departure delay, fall back to arrival delay
    delay_col = 'departure_delay_minutes' if 'departure_delay_minutes' in trip_filtered.columns else 'arrival_delay_minutes'

    # =============================================================================
    # STEP 6: Build edges using UNIFIED stops (consecutive stops within same trip)
    # =============================================================================
    trip_sorted = trip_filtered.sort_values(['trip_id', 'stop_sequence'])

    trip_sorted['next_stop'] = trip_sorted.groupby('trip_id')['unified_stop'].shift(-1)
    trip_sorted['next_delay'] = trip_sorted.groupby('trip_id')[delay_col].shift(-1)

    edges_raw = trip_sorted.dropna(subset=['next_stop'])

    edge_agg = (
        edges_raw
        .groupby(['unified_stop', 'next_stop'], as_index=False)
        .agg(
            mean_delay=(delay_col, 'mean'),
            num_trips=('trip_id', 'nunique'),
        )
        .rename(columns={'unified_stop': 'from_stop', 'next_stop': 'to_stop'})
    )

    print(f"Network edges (unified): {len(edge_agg):,}")

    # =============================================================================
    # STEP 7: Build UNIFIED node metadata
    # =============================================================================
    stop_meta = (
        trip_filtered
        .groupby('unified_stop', as_index=False)
        .agg(
            stop_name=('stop_name', 'first'),
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean'),
            num_platforms=('stop_point_ref', 'nunique'),
        )
    )

    # Filter to only stops with valid coordinates
    stop_meta = stop_meta[stop_meta['latitude'].notna() & stop_meta['longitude'].notna()]
    valid_stops = set(stop_meta['unified_stop'])

    # Filter edges to only include stops with valid coordinates on BOTH ends
    edge_agg = edge_agg[
        edge_agg['from_stop'].isin(valid_stops) & edge_agg['to_stop'].isin(valid_stops)
    ]

    print(f"Edges after filtering to valid coordinates: {len(edge_agg):,}")

    # Average delay per node (based on outgoing edges)
    node_delay = (
        edge_agg
        .groupby('from_stop')['mean_delay']
        .mean()
        .rename('avg_delay')
    )

    stop_meta = stop_meta.merge(
        node_delay, left_on='unified_stop', right_index=True, how='left'
    )

    return stop_meta, edge_agg


def build_network_graph(stop_meta: pd.DataFrame, edge_agg: pd.DataFrame) -> nx.DiGraph:
    """Build a NetworkX directed graph from stop metadata and edge aggregations."""
    print("\nBuilding NetworkX graph...")
    
    G = nx.DiGraph()

    for _, row in stop_meta.iterrows():
        label = row["stop_name"]
        if row['num_platforms'] > 1:
            label += f" ({int(row['num_platforms'])} platforms)"
        G.add_node(
            row['unified_stop'],
            label=label,
            x=float(row['longitude']),  # Use longitude as x
            y=float(row['latitude']),   # Use latitude as y
            latitude=float(row['latitude']),
            longitude=float(row['longitude']),
            avg_delay=float(row['avg_delay']) if pd.notna(row['avg_delay']) else 0.0,
            size=8,
        )

    for _, row in edge_agg.iterrows():
        G.add_edge(
            row['from_stop'],
            row['to_stop'],
            mean_delay=float(row['mean_delay']) if pd.notna(row['mean_delay']) else 0.0,
            num_trips=int(row['num_trips']),
        )

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    return G


def create_plotly_visualization(G: nx.Graph, output_path: Path):
    """Create interactive Plotly visualizations of the network with multiple styles."""
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    print(f"\nCreating Plotly visualizations...")
    
    # Extract node positions (using longitude as x, latitude as y)
    pos = {node: (data["longitude"], data["latitude"]) for node, data in G.nodes(data=True)}
    
    # Get all delays and compute percentile-based normalization
    all_delays = sorted([data.get("mean_delay", 0) for u, v, data in G.edges(data=True)])
    delay_min = min(all_delays)
    delay_max = max(all_delays)
    print(f"  Delay range: {delay_min:.2f} to {delay_max:.2f} min")
    print(f"  Median delay: {all_delays[len(all_delays)//2]:.2f} min")
    
    # Use RdYlGn_r colormap (Red-Yellow-Green reversed: green=low, red=high)
    cmap = plt.cm.RdYlGn_r
    
    # Use YlOrRd colormap for delays (yellow -> orange -> red)
    # Punctual (≤0.1 min) will be gray/neutral
    cmap_delayed = plt.cm.YlOrRd
    PUNCTUAL_THRESHOLD = 0.1  # minutes
    NEUTRAL_COLOR = "#888888"  # Gray for punctual
    
    # Get only delayed edges for percentile normalization
    delayed_values = sorted([d for d in all_delays if d > PUNCTUAL_THRESHOLD])
    
    def delay_to_color(delay):
        """Convert delay to color: gray for punctual, yellow-red gradient for delayed."""
        if delay <= PUNCTUAL_THRESHOLD:
            return NEUTRAL_COLOR
        
        # Percentile-based normalization among delayed edges only
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
                text=f"{G.nodes[u].get('label', u)} → {G.nodes[v].get('label', v)}<br>Avg delay: {delay:.1f} min<br>Trips: {data.get('num_trips', 0)}",
                showlegend=False,
            )
        )
    
    # Create node trace
    node_lon = [pos[node][0] for node in G.nodes()]
    node_lat = [pos[node][1] for node in G.nodes()]
    node_text = [f"{G.nodes[node].get('label', node)}<br>Avg delay: {G.nodes[node].get('avg_delay', 0):.1f} min" for node in G.nodes()]
    
    node_trace = go.Scattermapbox(
        lon=node_lon,
        lat=node_lat,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=8,
            color="#1f77b4",
        ),
    )
    
    # Create legend traces (invisible points with labels)
    legend_traces = [
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=NEUTRAL_COLOR),
            name="Punctual (≤0.1 min)",
            showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(0.0)[:3])),
            name="Slight delay",
            showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(0.5)[:3])),
            name="Moderate delay",
            showlegend=True,
        ),
        go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=10, color=mcolors.rgb2hex(cmap_delayed(1.0)[:3])),
            name="Significant delay",
            showlegend=True,
        ),
    ]
    
    # Generate both OSM and Positron styles
    styles = [
        ("open-street-map", "osm", "OpenStreetMap"),
        ("carto-positron", "positron", "Carto Positron"),
    ]
    
    for style_id, suffix, style_name in styles:
        fig = go.Figure(
            data=edge_traces + [node_trace] + legend_traces,
            layout=go.Layout(
                title=dict(
                    text=f"Tübingen Bus Network ({style_name})",
                    font=dict(size=20),
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    font=dict(size=12),
                ),
                hovermode="closest",
                mapbox=dict(
                    style=style_id,
                    center=dict(
                        lat=(TUEBINGEN_LAT_MIN + TUEBINGEN_LAT_MAX) / 2,
                        lon=(TUEBINGEN_LON_MIN + TUEBINGEN_LON_MAX) / 2,
                    ),
                    zoom=12,
                ),
                width=1400,
                height=900,
            ),
        )
        
        # Save to outputs folder
        style_output_path = output_path.parent / "outputs" / f"geo_network_{suffix}.html"
        style_output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(style_output_path))
        print(f"  Saved to {style_output_path}")
    
    return fig


def create_sigma_visualization(G: nx.Graph, output_path: Path):
    """Create an interactive ipysigma visualization of the network."""
    try:
        from ipysigma import Sigma
        
        print(f"\nCreating Sigma visualization...")
        
        # Create a copy with clean data types for JSON serialization
        G_clean = nx.DiGraph()
        
        for node, data in G.nodes(data=True):
            clean_data = {}
            for k, v in data.items():
                if hasattr(v, 'item'):  # numpy type
                    clean_data[k] = v.item()
                elif isinstance(v, (int, float, str, bool, type(None))):
                    clean_data[k] = v
                else:
                    clean_data[k] = float(v) if isinstance(v, (int, float)) else str(v)
            G_clean.add_node(node, **clean_data)
        
        for u, v, data in G.edges(data=True):
            clean_data = {}
            for k, val in data.items():
                if hasattr(val, 'item'):
                    clean_data[k] = val.item()
                elif isinstance(val, (int, float, str, bool, type(None))):
                    clean_data[k] = val
                else:
                    clean_data[k] = float(val) if isinstance(val, (int, float)) else str(val)
            G_clean.add_edge(u, v, **clean_data)
        
        sigma_path = str(output_path).replace(".html", "_sigma.html")
        Sigma.write_html(
            graph=G_clean,
            path=sigma_path,
            fullscreen=True,
            node_label="label",
            default_node_color="#DDDDDD",
            default_node_size=3,
            edge_color="mean_delay",
            edge_color_gradient="Reds",
            edge_color_scale="lin",
            edge_size="mean_delay",
            edge_size_range=(0.5, 4),
        )
        print(f"  Saved Sigma visualization to {sigma_path}")
        
    except ImportError:
        print("  ipysigma not installed, skipping Sigma visualization")
    except Exception as e:
        print(f"  Error creating Sigma visualization: {e}")


def create_matplotlib_visualization(G: nx.Graph, output_path: Path):
    """Create a static matplotlib visualization of the network."""
    import matplotlib.pyplot as plt
    
    print(f"\nCreating matplotlib visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Extract positions
    pos = {node: (data["longitude"], data["latitude"]) for node, data in G.nodes(data=True)}
    
    # Draw edges with color based on delay
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        delay = data.get("mean_delay", 0)
        if delay <= 0:
            color = "#00AA00"
        elif delay <= 2:
            color = "#CCCC00"
        elif delay <= 5:
            color = "#FF8800"
        else:
            color = "#FF0000"
        
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1, alpha=0.6, zorder=1)
    
    # Draw nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    ax.scatter(node_x, node_y, s=30, c="#1f77b4", edgecolors="white", linewidths=0.5, zorder=2)
    
    # Add labels for high-degree nodes
    for node in G.nodes():
        if G.degree(node) > 4:
            x, y = pos[node]
            label = G.nodes[node].get("label", node)
            # Shorten label
            label = label.replace("Tübingen ", "").split(" (")[0]
            ax.annotate(label, (x, y), fontsize=5, ha="center", va="bottom", 
                       xytext=(0, 3), textcoords="offset points")
    
    ax.set_title("Tübingen Bus Network (Geographic Coordinates)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#00AA00", linewidth=2, label="On time (≤0 min)"),
        Line2D([0], [0], color="#CCCC00", linewidth=2, label="Slight delay (≤2 min)"),
        Line2D([0], [0], color="#FF8800", linewidth=2, label="Moderate delay (≤5 min)"),
        Line2D([0], [0], color="#FF0000", linewidth=2, label="Significant delay (>5 min)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)
    
    # Save
    png_path = str(output_path).replace(".html", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {png_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("Tübingen Bus Network Graph Generator (Geographic Coordinates)")
    print("=" * 70)
    
    # Initialize GCS client
    bucket = init_gcs_client()
    
    # List and load trip files
    trip_files = list_trip_files(bucket)
    
    # Load trip data (limit files for faster processing if needed)
    # Set max_files=None to load all files, or a number like 100 for testing
    all_trip_data = download_and_combine_data(bucket, trip_files, max_files=None)
    
    if all_trip_data.empty:
        print("\nERROR: No trip data loaded.")
        return
    
    # Discover stop coordinates via TRIAS API
    coords_df = discover_stop_coordinates()
    
    # Process trip data to get stop metadata and edges
    stop_meta, edge_agg = process_trip_data(all_trip_data, coords_df)
    
    if stop_meta.empty:
        print("\nERROR: No stop metadata generated.")
        return
    
    # Build NetworkX graph
    G = build_network_graph(stop_meta, edge_agg)
    
    if G.number_of_nodes() == 0:
        print("\nERROR: No nodes in graph.")
        return
    
    # Create visualizations (OSM and Positron styles)
    create_plotly_visualization(G, OUTPUT_HTML)
    create_matplotlib_visualization(G, OUTPUT_HTML)
    
    print("\n" + "=" * 70)
    print("Done! Geographic network graph created successfully.")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
