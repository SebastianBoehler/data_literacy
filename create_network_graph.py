#!/usr/bin/env python3
"""
Network Graph Visualization for Tübingen Bus Network

Creates a network graph based on:
- network.csv: Grid-based coordinates of bus stops (matching Liniennetzplan layout)
- liniennetzplan_delays_by_line.csv: Connection data between stops with delay info

Output: Interactive HTML visualization using ipysigma/plotly
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import re

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
NETWORK_CSV = SCRIPT_DIR / "network.csv"
CONNECTIONS_CSV = SCRIPT_DIR / "liniennetzplan_delays_by_line.csv"
OUTPUT_HTML = SCRIPT_DIR / "liniennetzplan_network_graph.html"

# Line colors matching the Liniennetzplan (TüBus official colors)
LINE_COLORS = {
    "1": "#E30613",   # Red
    "2": "#009640",   # Green
    "3": "#0072BC",   # Blue
    "4": "#FFCC00",   # Yellow
    "5": "#E5007D",   # Magenta/Pink
    "6": "#00A0E3",   # Light Blue
    "7": "#F39200",   # Orange
    "8": "#95C11F",   # Light Green
    "X1": "#E30613",  # Express Red
    "X2": "#009640",  # Express Green
    "X15": "#E5007D", # Express Pink
    "N91": "#666666", # Night line
    "N92": "#666666",
    "N93": "#666666",
    "N94": "#666666",
    "N95": "#666666",
}
DEFAULT_LINE_COLOR = "#888888"


def normalize_stop_name(name: str) -> str:
    """
    Normalize stop names for matching between datasets.
    Handles variations like 'Tübingen Hauptbahnhof' vs 'Tübingen Hbf'.
    """
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove common prefixes for matching
    prefixes = ["Tübingen ", "Pfrondorf ", "Hagelloch ", "Hirschau ", 
                "Unterjesingen ", "Kilchberg ", "Weilheim ", "Bühl ", 
                "Bebenhausen ", "Kreßbach "]
    
    name_lower = name.lower()
    
    # Common abbreviation mappings
    replacements = {
        "str.": "straße",
        "str ": "straße ",
        "-str.": "-straße",
        "hbf": "hauptbahnhof",
        "bf": "bahnhof",
    }
    
    for old, new in replacements.items():
        name_lower = name_lower.replace(old, new)
    
    return name_lower


def load_network_coordinates() -> pd.DataFrame:
    """Load bus stop coordinates from network.csv"""
    print(f"Loading coordinates from {NETWORK_CSV}...")
    
    df = pd.read_csv(NETWORK_CSV, sep=";")
    df.columns = ["stop_name", "x", "y"]
    
    # Remove empty rows
    df = df.dropna(subset=["stop_name"])
    df = df[df["stop_name"].str.strip() != ""]
    
    # Convert coordinates to numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    
    # Create normalized name for matching
    df["stop_name_normalized"] = df["stop_name"].apply(normalize_stop_name)
    
    print(f"  Loaded {len(df)} stops with coordinates")
    return df


def load_connections() -> pd.DataFrame:
    """Load connection data from liniennetzplan_delays_by_line.csv"""
    print(f"Loading connections from {CONNECTIONS_CSV}...")
    
    df = pd.read_csv(CONNECTIONS_CSV)
    
    # Normalize stop names for matching
    df["stop_a_normalized"] = df["stop_a"].apply(normalize_stop_name)
    df["stop_b_normalized"] = df["stop_b"].apply(normalize_stop_name)
    
    print(f"  Loaded {len(df)} connections")
    print(f"  Unique lines: {df['line_name'].nunique()}")
    
    return df


def match_stop_to_coordinates(stop_name: str, coords_df: pd.DataFrame) -> dict | None:
    """
    Find matching coordinates for a stop name.
    Returns dict with x, y, original_name or None if not found.
    """
    stop_normalized = normalize_stop_name(stop_name)
    
    # Try exact match first
    exact_match = coords_df[coords_df["stop_name_normalized"] == stop_normalized]
    if len(exact_match) > 0:
        row = exact_match.iloc[0]
        return {"x": row["x"], "y": row["y"], "original_name": row["stop_name"]}
    
    # Try partial match (stop name contains the search term or vice versa)
    for _, row in coords_df.iterrows():
        if stop_normalized in row["stop_name_normalized"] or row["stop_name_normalized"] in stop_normalized:
            return {"x": row["x"], "y": row["y"], "original_name": row["stop_name"]}
    
    # Try fuzzy match on key parts
    stop_parts = stop_normalized.split()
    for _, row in coords_df.iterrows():
        coord_parts = row["stop_name_normalized"].split()
        # Check if significant words match
        common_words = set(stop_parts) & set(coord_parts)
        if len(common_words) >= 2:  # At least 2 words in common
            return {"x": row["x"], "y": row["y"], "original_name": row["stop_name"]}
    
    return None


def build_network_graph(coords_df: pd.DataFrame, connections_df: pd.DataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from coordinates and connections.
    """
    print("\nBuilding network graph...")
    
    G = nx.MultiGraph()  # MultiGraph to allow multiple edges (different lines) between same stops
    
    # Track which stops we've added and which connections we've made
    stops_added = set()
    stops_not_found = set()
    connections_made = 0
    connections_skipped = 0
    
    # Process each connection
    for _, row in connections_df.iterrows():
        stop_a = row["stop_a"]
        stop_b = row["stop_b"]
        line_name = str(row["line_name"])
        avg_delay = row.get("avg_delay", 0)
        n_obs = row.get("n_observations", 1)
        
        # Find coordinates for both stops
        coord_a = match_stop_to_coordinates(stop_a, coords_df)
        coord_b = match_stop_to_coordinates(stop_b, coords_df)
        
        if coord_a is None:
            stops_not_found.add(stop_a)
            connections_skipped += 1
            continue
        if coord_b is None:
            stops_not_found.add(stop_b)
            connections_skipped += 1
            continue
        
        # Use original names from coordinates file as node IDs
        node_a = coord_a["original_name"]
        node_b = coord_b["original_name"]
        
        # Add nodes if not already present
        if node_a not in stops_added:
            G.add_node(
                node_a,
                label=node_a,
                x=coord_a["x"],
                y=-coord_a["y"],  # Flip Y to match visual layout (top = high Y in image)
                size=8,
            )
            stops_added.add(node_a)
        
        if node_b not in stops_added:
            G.add_node(
                node_b,
                label=node_b,
                x=coord_b["x"],
                y=-coord_b["y"],
                size=8,
            )
            stops_added.add(node_b)
        
        # Add edge with line information
        line_color = LINE_COLORS.get(line_name, DEFAULT_LINE_COLOR)
        G.add_edge(
            node_a,
            node_b,
            line=line_name,
            color=line_color,
            avg_delay=avg_delay,
            n_observations=n_obs,
            weight=max(1, n_obs / 100),  # Edge weight based on frequency
        )
        connections_made += 1
    
    print(f"  Nodes added: {len(stops_added)}")
    print(f"  Edges added: {connections_made}")
    print(f"  Connections skipped (stops not found): {connections_skipped}")
    
    if stops_not_found:
        print(f"\n  Stops not found in coordinates ({len(stops_not_found)}):")
        for stop in sorted(stops_not_found)[:10]:
            print(f"    - {stop}")
        if len(stops_not_found) > 10:
            print(f"    ... and {len(stops_not_found) - 10} more")
    
    return G


def create_plotly_visualization(G: nx.Graph, output_path: Path):
    """Create an interactive Plotly visualization of the network."""
    import plotly.graph_objects as go
    
    print(f"\nCreating Plotly visualization...")
    
    # Extract node positions
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    
    # Create edge traces (one per line for coloring)
    edge_traces = []
    lines_in_graph = set()
    
    for u, v, data in G.edges(data=True):
        line_name = data.get("line", "unknown")
        lines_in_graph.add(line_name)
        color = data.get("color", DEFAULT_LINE_COLOR)
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=2, color=color),
                hoverinfo="text",
                text=f"Line {line_name}: {u} ↔ {v}<br>Avg delay: {data.get('avg_delay', 0):.1f} min",
                showlegend=False,
            )
        )
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[G.nodes[node].get("label", node) for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        marker=dict(
            size=10,
            color="#1f77b4",
            line=dict(width=1, color="white"),
        ),
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text="Tübingen Bus Network (Liniennetzplan Layout)",
                font=dict(size=20),
            ),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1400,
            height=900,
        ),
    )
    
    # Save to HTML
    fig.write_html(str(output_path))
    print(f"  Saved to {output_path}")
    
    return fig


def create_sigma_visualization(G: nx.Graph, output_path: Path):
    """Create an interactive ipysigma visualization of the network."""
    try:
        from ipysigma import Sigma
        
        print(f"\nCreating Sigma visualization...")
        
        # Convert MultiGraph to simple Graph for Sigma (keeping one edge per pair)
        G_simple = nx.Graph()
        
        for node, data in G.nodes(data=True):
            # Convert numpy types to Python native types for JSON serialization
            clean_data = {k: (int(v) if hasattr(v, 'item') else v) for k, v in data.items()}
            G_simple.add_node(node, **clean_data)
        
        # For edges, aggregate by stop pair
        edge_data = {}
        for u, v, data in G.edges(data=True):
            key = tuple(sorted([u, v]))
            if key not in edge_data:
                edge_data[key] = {
                    "lines": [],
                    "avg_delay": [],
                    "n_obs": 0,
                }
            edge_data[key]["lines"].append(data.get("line", "?"))
            edge_data[key]["avg_delay"].append(float(data.get("avg_delay", 0)))
            edge_data[key]["n_obs"] += int(data.get("n_observations", 1))
        
        for (u, v), data in edge_data.items():
            G_simple.add_edge(
                u, v,
                lines=", ".join(sorted(set(data["lines"]))),
                mean_delay=float(sum(data["avg_delay"]) / len(data["avg_delay"])) if data["avg_delay"] else 0.0,
                num_trips=int(data["n_obs"]),
            )
        
        Sigma.write_html(
            graph=G_simple,
            path=str(output_path).replace(".html", "_sigma.html"),
            fullscreen=True,
            node_label="label",
            default_node_color="#1f77b4",
            default_node_size=5,
            edge_color="mean_delay",
            edge_color_gradient="Reds",
            edge_color_scale="lin",
            edge_size="num_trips",
            edge_size_range=(0.5, 4),
        )
        print(f"  Saved Sigma visualization")
        
    except ImportError:
        print("  ipysigma not installed, skipping Sigma visualization")


def create_matplotlib_visualization(G: nx.Graph, output_path: Path):
    """Create a static matplotlib visualization of the network."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    print(f"\nCreating matplotlib visualization...")
    
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Extract positions
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    
    # Draw edges grouped by line
    lines_drawn = {}
    for u, v, data in G.edges(data=True):
        line_name = data.get("line", "unknown")
        color = data.get("color", DEFAULT_LINE_COLOR)
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2, alpha=0.7, zorder=1)
        lines_drawn[line_name] = color
    
    # Draw nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    ax.scatter(node_x, node_y, s=50, c="#1f77b4", edgecolors="white", linewidths=1, zorder=2)
    
    # Add labels for nodes with degree > 2 (important stops)
    for node in G.nodes():
        if G.degree(node) > 2:
            x, y = pos[node]
            label = node.replace("Tübingen ", "").replace("Pfrondorf ", "Pf. ")
            ax.annotate(label, (x, y), fontsize=6, ha="center", va="bottom", 
                       xytext=(0, 5), textcoords="offset points")
    
    # Create legend
    legend_patches = [mpatches.Patch(color=color, label=f"Line {line}") 
                     for line, color in sorted(lines_drawn.items())]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, ncol=2)
    
    ax.set_title("Tübingen Bus Network (Liniennetzplan Layout)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    # Save
    png_path = str(output_path).replace(".html", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {png_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("Tübingen Bus Network Graph Generator")
    print("=" * 70)
    
    # Load data
    coords_df = load_network_coordinates()
    connections_df = load_connections()
    
    # Build graph
    G = build_network_graph(coords_df, connections_df)
    
    if G.number_of_nodes() == 0:
        print("\nERROR: No nodes in graph. Check data files.")
        return
    
    # Create visualizations
    create_plotly_visualization(G, OUTPUT_HTML)
    create_matplotlib_visualization(G, OUTPUT_HTML)
    create_sigma_visualization(G, OUTPUT_HTML)
    
    print("\n" + "=" * 70)
    print("Done! Network graph created successfully.")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
