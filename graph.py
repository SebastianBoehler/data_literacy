from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from modules.trias import TriasClient
from modules.utils import ensure_directory, load_config, timestamp_slug

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

TRIP_CALLS_SOURCES: list[Path] = [SCRIPT_DIR / "exports"]
TRIP_CALLS_GLOB = "trip_calls_*.csv"

TRIAS_CONFIG_PATH = SCRIPT_DIR / "config.json"
FETCH_STOPS_FROM_TRIAS = True
STOPS_FALLBACK_CSV: Optional[Path] = None

OUTPUT_DIR = SCRIPT_DIR / "exports" / "graphs"
MIN_SEGMENTS = 1
FILE_PREFIX = "tuebingen_delay_graph"
GENERATE_PLOT = True
PLOT_FORMAT = "png"
NODE_LABEL_THRESHOLD: Optional[int] = 5
PLOT_FIG_SIZE = (14, 12)
POSITION_SCALE = 1.4

EARTH_RADIUS_KM = 6371.0

REQUIRED_TRIP_COLUMNS = {"journey_ref", "stop_point_ref", "stop_sequence"}
NUMERIC_TRIP_COLUMNS = (
    "stop_sequence",
    "arrival_delay_minutes",
    "departure_delay_minutes",
)


def discover_trip_call_files() -> list[Path]:
    files: list[Path] = []
    for source in TRIP_CALLS_SOURCES:
        path = Path(source)
        if not path.exists():
            continue
        if path.is_file():
            if path.match(TRIP_CALLS_GLOB):
                files.append(path)
            continue
        files.extend(sorted(path.glob(TRIP_CALLS_GLOB)))
    unique_files = []
    seen: set[Path] = set()
    for file_path in sorted(files):
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)
    return unique_files


def _base_stop_id(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    parts = str(ref).split(":")
    if len(parts) >= 3:
        return ":".join(parts[:3])
    return str(ref)


def _collect_stop_references(trip_calls: pd.DataFrame) -> list[str]:
    refs: set[str] = set()
    if "stop_point_ref" in trip_calls.columns:
        refs.update(
            str(value)
            for value in trip_calls["stop_point_ref"].dropna().astype(str).unique()
        )
    base_refs = {
        base
        for base in (
            _base_stop_id(value)
            for value in trip_calls.get(
                "stop_point_ref", pd.Series(dtype=str, index=trip_calls.index)
            )
        )
        if base is not None
    }
    refs.update(base_refs)
    return sorted(refs)


def _fetch_stops_from_trias(trip_calls: pd.DataFrame) -> pd.DataFrame:
    config = load_config(TRIAS_CONFIG_PATH)
    requestor_ref = config.get("trias_requestor_ref")
    if not requestor_ref:
        raise ValueError("TRIAS config missing 'trias_requestor_ref'.")

    client = TriasClient(requestor_ref)
    references = _collect_stop_references(trip_calls)
    if not references:
        return pd.DataFrame()

    stops = client.fetch_stop_details(references)
    if stops.empty:
        return stops

    stops = stops.copy()
    if "stop_point_ref" not in stops.columns:
        stops["stop_point_ref"] = pd.NA
    stops["stop_point_ref"] = stops["stop_point_ref"].fillna(stops.get("trias_ref"))
    stops["base_stop_id"] = stops.get("stop_id")
    if stops["base_stop_id"].isna().any():
        stops.loc[stops["base_stop_id"].isna(), "base_stop_id"] = stops.loc[
            stops["base_stop_id"].isna(), "stop_point_ref"
        ].map(_base_stop_id)

    keep_columns = [
        col
        for col in (
            "stop_point_ref",
            "stop_id",
            "base_stop_id",
            "stop_name",
            "latitude",
            "longitude",
        )
        if col in stops.columns
    ]
    stops = (
        stops[keep_columns]
        .dropna(subset=["stop_point_ref"])
        .drop_duplicates(subset=["stop_point_ref"], keep="first")
        .reset_index(drop=True)
    )
    return stops


def load_stop_metadata_for_graph(trip_calls: pd.DataFrame) -> pd.DataFrame:
    stops = pd.DataFrame()
    if FETCH_STOPS_FROM_TRIAS:
        try:
            stops = _fetch_stops_from_trias(trip_calls)
        except Exception as exc:
            print(f"Failed to fetch stop metadata from TRIAS: {exc}")
            stops = pd.DataFrame()

    if (stops.empty or "stop_point_ref" not in stops.columns) and STOPS_FALLBACK_CSV:
        try:
            stops = load_stop_metadata(STOPS_FALLBACK_CSV)
            if "stop_point_ref" not in stops.columns:
                stops["stop_point_ref"] = stops.get("stop_id")
        except FileNotFoundError:
            print(f"Stop fallback CSV not found at {STOPS_FALLBACK_CSV}")
            stops = pd.DataFrame()

    if stops.empty:
        return stops

    if "base_stop_id" not in stops.columns:
        stops["base_stop_id"] = stops["stop_point_ref"].map(_base_stop_id)
    return stops


def load_trip_calls(csv_path: Path) -> pd.DataFrame:
    """Load and clean trip call exports from TRIAS."""

    trip_calls = pd.read_csv(csv_path)
    if trip_calls.empty:
        raise ValueError("Trip call CSV is empty. Did you fetch trip data first?")

    missing = REQUIRED_TRIP_COLUMNS.difference(trip_calls.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Trip call CSV is missing required columns: {missing_list}.\n"
            "Make sure you passed the detailed export from trip.py."
        )

    for column in NUMERIC_TRIP_COLUMNS:
        if column in trip_calls.columns:
            trip_calls[column] = pd.to_numeric(trip_calls[column], errors="coerce")

    # Remove rows that cannot be used to build an ordered journey sequence.
    trip_calls = trip_calls.dropna(
        subset=["journey_ref", "stop_point_ref", "stop_sequence"]
    ).copy()

    trip_calls["stop_sequence"] = trip_calls["stop_sequence"].astype(int)
    trip_calls["journey_ref"] = trip_calls["journey_ref"].astype(str)

    if "operating_day_ref" not in trip_calls.columns:
        trip_calls["operating_day_ref"] = ""

    trip_calls["operating_day_ref"] = trip_calls["operating_day_ref"].astype(str)
    trip_calls["journey_id"] = (
        trip_calls["journey_ref"] + "|" + trip_calls["operating_day_ref"]
    )

    order_columns = [
        "journey_id",
        "stop_sequence",
    ]
    if "arrival_planned" in trip_calls.columns:
        trip_calls["arrival_planned"] = pd.to_datetime(
            trip_calls["arrival_planned"], errors="coerce"
        )
        order_columns.append("arrival_planned")

    trip_calls = trip_calls.sort_values(order_columns).reset_index(drop=True)
    return trip_calls


def load_all_trip_calls() -> pd.DataFrame:
    files = discover_trip_call_files()
    if not files:
        raise FileNotFoundError(
            "No trip call CSV files found. Configure TRIP_CALLS_SOURCES to point to exports."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in files:
        frame = load_trip_calls(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)
        print(f"Loaded {len(frame)} trip call rows from {csv_path}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.sort_values(["journey_id", "stop_sequence"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"Aggregated {len(combined)} trip call rows across {len(files)} files")
    return combined


def load_stop_metadata(csv_path: Path) -> pd.DataFrame:
    """Load stop metadata to obtain coordinates for plotting."""

    stops = pd.read_csv(csv_path)
    if stops.empty:
        raise ValueError("Stop metadata CSV is empty.")

    if "stop_point_ref" not in stops.columns:
        if "stop_id" not in stops.columns:
            raise ValueError(
                "Stop metadata must contain either 'stop_point_ref' or 'stop_id'."
            )
        stops["stop_point_ref"] = stops["stop_id"]

    keep_columns = [
        col
        for col in ("stop_point_ref", "stop_name", "latitude", "longitude")
        if col in stops.columns
    ]
    metadata = stops[keep_columns].dropna(subset=["stop_point_ref"]).copy()
    metadata = metadata.drop_duplicates(subset="stop_point_ref", keep="first")
    return metadata


def _most_common(series: pd.Series) -> Optional[str]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode()
    if not modes.empty:
        return modes.iloc[0]
    return non_null.iloc[0]


def summarize_nodes(trip_calls: pd.DataFrame) -> pd.DataFrame:
    grouped = trip_calls.groupby("stop_point_ref", sort=False)
    summary = grouped.agg(
        stop_name=("stop_name", _most_common),
        call_count=("stop_point_ref", "size"),
        avg_arrival_delay=("arrival_delay_minutes", "mean"),
        avg_departure_delay=("departure_delay_minutes", "mean"),
        median_arrival_delay=("arrival_delay_minutes", "median"),
        median_departure_delay=("departure_delay_minutes", "median"),
    ).reset_index()

    for column in (
        "avg_arrival_delay",
        "avg_departure_delay",
        "median_arrival_delay",
        "median_departure_delay",
    ):
        if column in summary.columns:
            summary[column] = summary[column].round(3)

    return summary


def summarize_edges(trip_calls: pd.DataFrame) -> pd.DataFrame:
    edge_frame = trip_calls[
        [
            "journey_id",
            "journey_ref",
            "operating_day_ref",
            "stop_point_ref",
            "stop_name",
            "stop_sequence",
            "arrival_delay_minutes",
            "departure_delay_minutes",
        ]
    ].copy()

    edge_frame["next_stop_point_ref"] = edge_frame.groupby("journey_id")[
        "stop_point_ref"
    ].shift(-1)
    edge_frame["next_stop_name"] = edge_frame.groupby("journey_id")[
        "stop_name"
    ].shift(-1)
    edge_frame["next_arrival_delay"] = edge_frame.groupby("journey_id")[
        "arrival_delay_minutes"
    ].shift(-1)
    edge_frame["next_departure_delay"] = edge_frame.groupby("journey_id")[
        "departure_delay_minutes"
    ].shift(-1)

    edges = edge_frame.dropna(subset=["next_stop_point_ref"])
    edges = edges[edges["stop_point_ref"] != edges["next_stop_point_ref"]].copy()

    edges["delay_delta"] = (
        edges["next_arrival_delay"] - edges["arrival_delay_minutes"]
    )

    if "operating_day_ref" in edges.columns:
        edges["journey_key"] = (
            edges["journey_ref"].astype(str)
            + "|"
            + edges["operating_day_ref"].astype(str)
        )
    else:
        edges["journey_key"] = edges["journey_ref"].astype(str)

    summary = (
        edges.groupby(["stop_point_ref", "next_stop_point_ref"], sort=False)
        .agg(
            segment_count=("journey_id", "size"),
            journey_count=("journey_key", pd.Series.nunique),
            avg_delay_delta=("delay_delta", "mean"),
            avg_arrival_delay_next=("next_arrival_delay", "mean"),
            avg_arrival_delay_current=("arrival_delay_minutes", "mean"),
        )
        .reset_index()
    )

    numeric_columns = (
        "segment_count",
        "journey_count",
        "avg_delay_delta",
        "avg_arrival_delay_next",
        "avg_arrival_delay_current",
    )
    for column in numeric_columns:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    summary["avg_delay_delta"] = summary["avg_delay_delta"].round(3)
    summary["avg_arrival_delay_next"] = summary["avg_arrival_delay_next"].round(3)
    summary["avg_arrival_delay_current"] = summary["avg_arrival_delay_current"].round(3)

    summary.rename(
        columns={
            "stop_point_ref": "source",
            "next_stop_point_ref": "target",
        },
        inplace=True,
    )
    summary["weight"] = summary["avg_delay_delta"].fillna(
        summary["avg_arrival_delay_next"]
    )
    summary["weight"] = summary["weight"].fillna(0.0)
    summary["weight"] = summary["weight"].round(3)

    summary["avg_delay_delta_positive"] = (
        summary["avg_delay_delta"].clip(lower=0.0).round(3)
    )

    return summary


def build_delay_graph(
    trip_calls: pd.DataFrame,
    stops: Optional[pd.DataFrame] = None,
    *,
    min_segments: int = 1,
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Create a weighted directed graph from trip call observations."""

    nodes_df = summarize_nodes(trip_calls)
    if stops is not None and not stops.empty:
        stops_map = stops.set_index("stop_point_ref")
        if "latitude" in stops_map.columns:
            nodes_df["latitude"] = nodes_df["stop_point_ref"].map(
                stops_map["latitude"]
            )
        if "longitude" in stops_map.columns:
            nodes_df["longitude"] = nodes_df["stop_point_ref"].map(
                stops_map["longitude"]
            )
        if "stop_name" in stops_map.columns:
            nodes_df["stop_name"] = nodes_df["stop_name"].fillna(
                nodes_df["stop_point_ref"].map(stops_map["stop_name"])
            )
    else:
        nodes_df["latitude"] = None
        nodes_df["longitude"] = None

    edges_df = summarize_edges(trip_calls)
    if min_segments > 1:
        edges_df = edges_df[edges_df["segment_count"] >= min_segments].reset_index(
            drop=True
        )

    graph = nx.DiGraph()

    for row in nodes_df.itertuples(index=False):
        node_attributes: dict[str, object] = {
            "stop_name": row.stop_name,
            "call_count": int(row.call_count),
            "avg_arrival_delay": _safe_float(row.avg_arrival_delay, default=0.0),
            "avg_departure_delay": _safe_float(row.avg_departure_delay, default=0.0),
            "median_arrival_delay": _safe_float(row.median_arrival_delay, default=0.0),
            "median_departure_delay": _safe_float(row.median_departure_delay, default=0.0),
        }

        if hasattr(row, "latitude") and not pd.isna(row.latitude):
            node_attributes["latitude"] = float(row.latitude)
        if hasattr(row, "longitude") and not pd.isna(row.longitude):
            node_attributes["longitude"] = float(row.longitude)

        cleaned_attributes = {
            key: value for key, value in node_attributes.items() if value is not None
        }

        graph.add_node(row.stop_point_ref, **cleaned_attributes)

    for row in edges_df.itertuples(index=False):
        edge_attributes: dict[str, object] = {
            "segment_count": int(row.segment_count),
            "journey_count": int(row.journey_count),
            "avg_delay_delta": _safe_float(row.avg_delay_delta, default=0.0),
            "avg_arrival_delay_next": _safe_float(row.avg_arrival_delay_next, default=0.0),
            "avg_arrival_delay_current": _safe_float(row.avg_arrival_delay_current, default=0.0),
            "avg_delay_delta_positive": _safe_float(row.avg_delay_delta_positive, default=0.0),
            "weight": _safe_float(row.weight, default=0.0),
        }

        cleaned_edge_attributes = {
            key: value for key, value in edge_attributes.items() if value is not None
        }

        graph.add_edge(row.source, row.target, **cleaned_edge_attributes)

    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    graph.graph["generated_from"] = "trip_calls"

    return graph, nodes_df, edges_df


def plot_graph(graph: nx.DiGraph, output_path: Path) -> None:
    """Visualise the graph and save it as an image."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot plot an empty graph.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or drop the --plot flag."
        ) from exc

    plt.switch_backend("Agg")

    positions = _resolve_positions(graph)

    delays = [
        _safe_float(attr.get("avg_arrival_delay"), default=0.0)
        for _, attr in graph.nodes(data=True)
    ]
    min_delay = min(delays) if delays else 0.0
    max_delay = max(delays) if delays else 0.0
    if min_delay == max_delay:
        min_delay -= 1.0
        max_delay += 1.0

    cmap = plt.colormaps.get("RdYlGn_r")
    norm = plt.Normalize(vmin=min_delay, vmax=max_delay)
    node_colors = [cmap(norm(delay)) for delay in delays]

    edge_segment_counts = [
        attr.get("segment_count", 1) for _, _, attr in graph.edges(data=True)
    ]
    max_segment = max(edge_segment_counts) if edge_segment_counts else 1
    edge_widths = [0.5 + (count / max_segment) * 3 for count in edge_segment_counts]

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color="#777777",
        width=edge_widths,
        arrows=True,
        arrowsize=12,
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=[
            200 + attr.get("call_count", 1) * 10 for _, attr in graph.nodes(data=True)
        ],
    )
    label_candidates = {}
    for node, data in graph.nodes(data=True):
        call_count = data.get("call_count", 0)
        if NODE_LABEL_THRESHOLD is None or call_count >= NODE_LABEL_THRESHOLD:
            label_candidates[node] = data.get("stop_name", node)

    if label_candidates:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=label_candidates,
            font_size=8,
            ax=ax,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average arrival delay (minutes)")

    ax.set_title("Tübingen delay propagation graph")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _resolve_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    valid_nodes: list[tuple[str, float, float]] = []
    for node, data in graph.nodes(data=True):
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue
        valid_nodes.append((str(node), float(lat), float(lon)))

    if valid_nodes:
        lats = [lat for _, lat, _ in valid_nodes]
        lons = [lon for _, _, lon in valid_nodes]
        lat_mean = sum(lats) / len(lats)
        lon_mean = sum(lons) / len(lons)
        cos_lat = math.cos(math.radians(lat_mean)) or 1.0

        raw_coords: dict[str, tuple[float, float]] = {}
        for node, lat, lon in valid_nodes:
            x = EARTH_RADIUS_KM * math.radians(lon - lon_mean) * cos_lat
            y = EARTH_RADIUS_KM * math.radians(lat - lat_mean)
            raw_coords[node] = (x, y)

        buckets: dict[tuple[int, int], list[str]] = {}
        for node, (x, y) in raw_coords.items():
            x *= POSITION_SCALE
            y *= POSITION_SCALE
            key = (round(x, 3), round(y, 3))
            buckets.setdefault(key, []).append(node)

        jitter_radius = 0.05
        for key, nodes in buckets.items():
            base_x, base_y = key
            if len(nodes) == 1:
                coords[nodes[0]] = (base_x, base_y)
                continue
            for index, node in enumerate(nodes):
                angle = 2 * math.pi * index / len(nodes)
                jitter_x = base_x + jitter_radius * POSITION_SCALE * math.cos(angle)
                jitter_y = base_y + jitter_radius * POSITION_SCALE * math.sin(angle)
                coords[node] = (jitter_x, jitter_y)

        if coords:
            return coords

    return nx.spring_layout(graph, seed=42)


def _safe_float(value: object, *, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv:
        raise ValueError("This script no longer accepts command-line arguments. Adjust constants instead.")

    trip_calls = load_all_trip_calls()
    stops = load_stop_metadata_for_graph(trip_calls)

    graph, nodes_df, edges_df = build_delay_graph(
        trip_calls,
        stops,
        min_segments=max(1, MIN_SEGMENTS),
    )

    output_dir = ensure_directory(OUTPUT_DIR)
    timestamp = timestamp_slug()
    prefix = f"{FILE_PREFIX}_{timestamp}"

    nodes_path = output_dir / f"{prefix}_nodes.csv"
    edges_path = output_dir / f"{prefix}_edges.csv"
    gexf_path = output_dir / f"{prefix}.gexf"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    nx.write_gexf(graph, gexf_path)

    print(f"Saved node metrics to {nodes_path}")
    print(f"Saved edge metrics to {edges_path}")
    print(f"Saved graph structure to {gexf_path}")

    if GENERATE_PLOT:
        plot_path = output_dir / f"{prefix}.{PLOT_FORMAT}"
        plot_graph(graph, plot_path)
        print(f"Saved graph visualisation to {plot_path}")


if __name__ == "__main__":
    main()
