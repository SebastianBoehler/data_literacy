"""
Discover all bus stops in Tübingen area using TRIAS API.
Uses multiple center points to ensure full coverage.

Outputs:
- outputs/all_stops_coordinates.csv - All discovered stops with coordinates
- Updates outputs/all_trip_data.parquet with missing coordinates
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from modules.trias import TriasClient

SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
TRIP_DATA_PATH = OUTPUT_DIR / "all_trip_data.parquet"

# TRIAS API key (same as used in create_geo_network_graph.py)
REQUESTOR_REF = os.environ.get("TRIAS_API_KEY", "SeBaSTiaN_BoeHLeR")

# Center points to search from (covering Tübingen area)
# Start with main station, then expand to cover suburbs
SEARCH_CENTERS = [
    # Core Tübingen
    (48.5156, 9.0575, "Hauptbahnhof"),
    (48.5220, 9.0520, "Altstadt"),
    (48.5100, 9.0700, "Südstadt"),
    (48.5300, 9.0600, "Nordstadt"),
    (48.5200, 9.0300, "Weststadt"),
    (48.5350, 9.0450, "WHO/Schnarrenberg"),
    (48.4950, 9.0500, "Derendingen"),
    (48.5400, 9.0800, "Waldhäuser Ost"),
    # Lustnau area (Line 22)
    (48.5150, 9.0900, "Lustnau"),
    (48.5200, 9.0850, "Lustnau Mitte"),
    (48.5250, 9.0950, "Lustnau Nord"),
    (48.5100, 9.0850, "Lustnau Süd"),
    (48.5180, 9.1050, "Gartenstraße area"),
    # Bebenhausen
    (48.5600, 9.0600, "Bebenhausen"),
    # Pfrondorf
    (48.5500, 9.0200, "Pfrondorf"),
    # Hagelloch
    (48.5450, 9.0100, "Hagelloch"),
    # Hirschau / Kilchberg
    (48.4850, 9.0600, "Hirschau"),
    (48.4700, 9.0400, "Kilchberg"),
    # Wanne / Sand
    (48.5000, 9.0800, "Wanne"),
    (48.4900, 9.0700, "Sand"),
    # Surrounding towns
    (48.4500, 9.0500, "Rottenburg"),
    (48.4900, 9.2100, "Reutlingen"),
    (48.4000, 9.0600, "Mössingen"),
    (48.4600, 9.1500, "Gomaringen"),
    (48.5800, 9.0000, "Dettenhausen"),
    (48.5300, 8.9500, "Waldenbuch"),
    (48.5700, 9.1000, "Kirchentellinsfurt"),
    (48.5400, 9.1500, "Kusterdingen/Wankheim"),
    (48.4800, 9.1200, "Ofterdingen"),
    (48.5600, 8.9000, "Steinenbronn"),
    (48.6800, 9.2200, "Echterdingen/Flughafen"),
    (48.4300, 9.2000, "Betzingen"),
    # Weilheim / Mähringen
    (48.5200, 9.1300, "Weilheim"),
    (48.5400, 9.1200, "Mähringen"),
    # Additional Tübingen neighborhoods
    (48.5050, 9.0400, "Südstadt West"),
    (48.5250, 9.0700, "Österberg"),
    (48.5350, 9.0700, "Feuerhägle"),
    (48.5150, 9.0400, "Französisches Viertel"),
    (48.5080, 9.0550, "Loretto"),
]

SEARCH_RADIUS_KM = 5.0  # 5km radius per search for better coverage

# NOTE: LIMITATION - Some Tübingen stops are not discoverable via TRIAS API
# The following stops have records in our data but cannot be found via the API:
# - Tübingen WHO Ahornweg (4,814 records)
# - Tübingen WHO Ulmenweg (3,649 records)
# - Tübingen Botanischer Garten (1,419 records)
# - Tübingen BG Unfallklinik (2,036 records)
# - Tübingen Uni-Kliniken Berg (2,034 records)
# - Tübingen Neuhalde (32 records)
# - Tübingen Nürtinger Straße (28 records)
# These stops appear in trip data but TRIAS API returns them under different names
# or doesn't include them in radius searches. This affects ~14,000 records (~10%).


def discover_stops_from_centers(client, centers, all_stops, radius_km=SEARCH_RADIUS_KM):
    """Discover stops from a list of center points."""
    new_stops = {}
    
    for lat, lon, name in centers:
        print(f"\nSearching from {name} ({lat:.4f}, {lon:.4f})...")
        
        try:
            stops_df = client.fetch_stops(
                center=(lat, lon),
                radius_km=radius_km,
                max_results=500
            )
            
            found_new = 0
            for _, row in stops_df.iterrows():
                stop_name = row.get('stop_name') or row.get('name')
                stop_lat = row.get('latitude') or row.get('lat')
                stop_lon = row.get('longitude') or row.get('lon')
                stop_ref = row.get('stop_point_ref') or row.get('stop_ref')
                
                if stop_name and pd.notna(stop_lat) and pd.notna(stop_lon):
                    if stop_name not in all_stops and stop_name not in new_stops:
                        new_stops[stop_name] = {
                            'stop_name': stop_name,
                            'latitude': float(stop_lat),
                            'longitude': float(stop_lon),
                            'stop_ref': stop_ref
                        }
                        found_new += 1
            
            print(f"  Found {len(stops_df)} stops ({found_new} new)")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return new_stops


def discover_stops():
    """Discover all stops using iterative expansion from discovered stops."""
    
    if not REQUESTOR_REF:
        print("ERROR: TRIAS_API_KEY environment variable not set")
        print("Set it with: export TRIAS_API_KEY='your_key'")
        return None
    
    client = TriasClient(REQUESTOR_REF)
    
    all_stops = {}  # stop_name -> {lat, lon, stop_ref}
    
    # Phase 1: Initial discovery from predefined centers
    print("=" * 60)
    print("PHASE 1: Initial discovery from predefined centers")
    print("=" * 60)
    
    new_stops = discover_stops_from_centers(client, SEARCH_CENTERS, all_stops)
    all_stops.update(new_stops)
    print(f"\nAfter Phase 1: {len(all_stops)} stops discovered")
    
    # Phase 2: Iterative expansion - use discovered stops as new centers
    iteration = 0
    max_iterations = 5  # Limit iterations to avoid infinite loops
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"PHASE 2 - Iteration {iteration}: Expanding from discovered stops")
        print("=" * 60)
        
        # Use stops at the edges (furthest from Tübingen center) as new search centers
        tub_center = (48.5156, 9.0575)
        
        # Calculate distance from center for each stop
        stops_with_dist = []
        for name, data in all_stops.items():
            dist = ((data['latitude'] - tub_center[0])**2 + (data['longitude'] - tub_center[1])**2)**0.5
            stops_with_dist.append((name, data, dist))
        
        # Sort by distance and take the furthest ones as new centers
        stops_with_dist.sort(key=lambda x: x[2], reverse=True)
        
        # Take top 20 furthest stops as new search centers
        new_centers = [
            (data['latitude'], data['longitude'], name[:30])
            for name, data, _ in stops_with_dist[:20]
        ]
        
        # Also add some random stops from middle distances
        mid_stops = stops_with_dist[len(stops_with_dist)//3:2*len(stops_with_dist)//3]
        import random
        random.seed(iteration)
        if len(mid_stops) > 10:
            sampled = random.sample(mid_stops, 10)
            new_centers.extend([
                (data['latitude'], data['longitude'], name[:30])
                for name, data, _ in sampled
            ])
        
        # Discover from new centers
        new_stops = discover_stops_from_centers(client, new_centers, all_stops, radius_km=3.0)
        
        if not new_stops:
            print(f"\nNo new stops found in iteration {iteration}. Stopping.")
            break
        
        all_stops.update(new_stops)
        print(f"\nAfter iteration {iteration}: {len(all_stops)} total stops (+{len(new_stops)} new)")
    
    print(f"\n{'='*60}")
    print(f"DISCOVERY COMPLETE: {len(all_stops)} unique stops")
    print("=" * 60)
    
    # Convert to DataFrame
    stops_df = pd.DataFrame(list(all_stops.values()))
    return stops_df


def update_trip_data(stops_df: pd.DataFrame):
    """Update trip data with missing coordinates."""
    
    if not TRIP_DATA_PATH.exists():
        print(f"ERROR: {TRIP_DATA_PATH} not found")
        return
    
    print(f"\nLoading trip data from {TRIP_DATA_PATH}...")
    trip_df = pd.read_parquet(TRIP_DATA_PATH)
    print(f"Loaded {len(trip_df):,} records")
    
    # Check current coordinate coverage
    has_coords = trip_df['latitude'].notna() & trip_df['longitude'].notna()
    print(f"Records with coordinates: {has_coords.sum():,} ({has_coords.sum()/len(trip_df)*100:.1f}%)")
    print(f"Records missing coordinates: {(~has_coords).sum():,}")
    
    # Create lookup dict from discovered stops
    coord_lookup = {}
    for _, row in stops_df.iterrows():
        coord_lookup[row['stop_name']] = (row['latitude'], row['longitude'])
    
    # Fill missing coordinates
    filled_count = 0
    for idx in trip_df[~has_coords].index:
        stop_name = trip_df.loc[idx, 'stop_name']
        if stop_name in coord_lookup:
            trip_df.loc[idx, 'latitude'] = coord_lookup[stop_name][0]
            trip_df.loc[idx, 'longitude'] = coord_lookup[stop_name][1]
            filled_count += 1
    
    print(f"\nFilled {filled_count:,} records with coordinates")
    
    # Check new coverage
    has_coords_new = trip_df['latitude'].notna() & trip_df['longitude'].notna()
    print(f"New coverage: {has_coords_new.sum():,} ({has_coords_new.sum()/len(trip_df)*100:.1f}%)")
    
    # Find stops still missing coordinates
    still_missing = trip_df[~has_coords_new]['stop_name'].unique()
    if len(still_missing) > 0:
        print(f"\nStops still missing coordinates ({len(still_missing)}):")
        for s in sorted(still_missing):
            print(f"  - {s}")
    
    # Save updated data
    backup_path = OUTPUT_DIR / "all_trip_data_backup.parquet"
    print(f"\nBacking up original to {backup_path}...")
    pd.read_parquet(TRIP_DATA_PATH).to_parquet(backup_path)
    
    print(f"Saving updated data to {TRIP_DATA_PATH}...")
    trip_df.to_parquet(TRIP_DATA_PATH)
    print("Done!")
    
    return trip_df


def main():
    print("=" * 60)
    print("TRIAS Stop Discovery - Fill Missing Coordinates")
    print("=" * 60)
    
    # Discover stops
    stops_df = discover_stops()
    
    if stops_df is None or stops_df.empty:
        print("No stops discovered, exiting")
        return
    
    # Save discovered stops
    stops_path = OUTPUT_DIR / "all_stops_coordinates.csv"
    stops_df.to_csv(stops_path, index=False)
    print(f"\nSaved all stops to {stops_path}")
    
    # Update trip data
    update_trip_data(stops_df)


if __name__ == "__main__":
    main()
