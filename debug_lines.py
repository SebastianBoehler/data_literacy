#!/usr/bin/env python3
"""
Debug script to analyze all unique lines in the departure data.
Identifies trains, regional express, and other non-bus lines for filtering.
"""

import os
import io
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'departure-data-reader-key.json'

client = storage.Client()
bucket = client.get_bucket('departure_data')

# Get departure files
blobs = list(bucket.list_blobs())
departure_files = [b.name for b in blobs if 'departures' in b.name and b.name.endswith('.csv')]
print(f"Found {len(departure_files)} departure files")

# Load a sample (first 50 files for speed)
sample_files = departure_files[:50]

def load_file(file_name):
    blob = bucket.blob(file_name)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

print(f"Loading {len(sample_files)} sample files...")
dfs = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(load_file, f): f for f in sample_files}
    for future in as_completed(futures):
        try:
            dfs.append(future.result())
        except Exception as e:
            print(f"Error: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows")

# Analyze unique lines
line_stats = df.groupby('line_name').agg(
    count=('line_name', 'size'),
    destinations=('destination', lambda x: list(x.dropna().unique()[:5])),
    stops=('stop_name', lambda x: list(x.dropna().unique()[:3]))
).reset_index().sort_values('count', ascending=False)

print("\n" + "=" * 80)
print("ALL UNIQUE LINES")
print("=" * 80)

for _, row in line_stats.iterrows():
    line = row['line_name']
    count = row['count']
    dests = row['destinations']
    stops = row['stops']
    print(f"\nLine: {line!r} ({count:,} departures)")
    print(f"  Destinations: {dests}")
    print(f"  Sample stops: {stops}")

# Categorize
print("\n" + "=" * 80)
print("CATEGORIZATION SUGGESTIONS")
print("=" * 80)

trains = []
buses = []
other = []

for line in line_stats['line_name'].unique():
    line_str = str(line).strip()
    
    # Train patterns
    if re.match(r'^(RE|RB|IRE|IC|ICE|S\d)', line_str, re.IGNORECASE):
        trains.append(line_str)
    elif re.match(r'^\d{1,3}$', line_str):  # Numeric lines 1-999
        buses.append(line_str)
    else:
        other.append(line_str)

print(f"\nLikely TRAINS/REGIONAL ({len(trains)}): {sorted(trains)}")
print(f"\nLikely BUSES ({len(buses)}): {sorted(buses, key=lambda x: int(x) if x.isdigit() else 999)}")
print(f"\nOTHER/UNCLEAR ({len(other)}): {sorted(other)}")

# Export for reference
line_stats.to_csv('line_analysis.csv', index=False)
print("\nSaved detailed analysis to 'line_analysis.csv'")
