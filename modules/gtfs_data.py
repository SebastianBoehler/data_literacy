#!/usr/bin/env python3
"""
GTFS Data Module

Extract and process public transport data from GTFS API.
Focus on Tübingen stops with coordinates for weather integration.

Author: Data Literacy Project - University of Tübingen
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

# Optional imports
try:
    import geopandas as gpd
    from shapely import wkt
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    gpd = None
    wkt = None

logger = logging.getLogger(__name__)

class GTFSDataExtractor:
    """Extract and process GTFS data for Tübingen transport analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize GTFS data extractor.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config
        self.base_url = config.get('gtfs_api_url', 'https://api.mobidata-bw.de/gtfs/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('user_agent', 'Data-Literacy-Project-University-Tuebingen/1.0')
        })
        
        # Tübingen municipality boundaries - read from config or use defaults
        self.tuebingen_bounds = config.get('tuebingen_bounds', {
            'min_lat': 48.480,
            'max_lat': 48.570,
            'min_lon': 9.000,
            'max_lon': 9.130
        })
    
    def fetch_gtfs_table(self, table_name: str) -> pd.DataFrame:
        """
        Fetch a specific GTFS table from the API.
        
        Args:
            table_name: Name of the GTFS table (stops, routes, trips, etc.)
            
        Returns:
            DataFrame with GTFS data
        """
        url = f"{self.base_url}{table_name}"
        
        try:
            logger.info(f"Fetching GTFS table: {table_name}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(response.text)
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {table_name}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error parsing {table_name}: {e}")
            return pd.DataFrame()
    
    def extract_coordinates_from_geometry(self, geometry_str: str) -> Optional[tuple]:
        """
        Extract latitude and longitude from PostGIS POINT geometry.
        
        Args:
            geometry_str: PostGIS POINT string (e.g., "POINT(9.059184 48.522302)")
            
        Returns:
            Tuple of (longitude, latitude) or None if parsing fails
        """
        try:
            if pd.isna(geometry_str) or not geometry_str:
                return None
            
            # Parse PostGIS POINT format
            if geometry_str.startswith('POINT('):
                coords = geometry_str.replace('POINT(', '').replace(')', '')
                lon, lat = map(float, coords.split())
                return lon, lat
            
            return None
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse geometry {geometry_str}: {e}")
            return None
    
    def filter_tuebingen_stops(self, stops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stops to include only Tübingen municipality.
        
        Args:
            stops_df: DataFrame with all stops
            
        Returns:
            DataFrame filtered to Tübingen stops only
        """
        if stops_df.empty:
            return stops_df
        
        # Extract coordinates
        coords = stops_df['stop_geometry'].apply(self.extract_coordinates_from_geometry)
        valid_coords = coords.dropna()
        
        if len(valid_coords) == 0:
            logger.warning("No valid coordinates found in stops data")
            return pd.DataFrame()
        
        # Create separate lat/lon columns
        stops_df = stops_df.copy()
        stops_df['longitude'] = coords.apply(lambda x: x[0] if x else None)
        stops_df['latitude'] = coords.apply(lambda x: x[1] if x else None)
        
        # Filter by Tübingen boundaries
        tuebingen_mask = (
            (stops_df['latitude'] >= self.tuebingen_bounds['min_lat']) &
            (stops_df['latitude'] <= self.tuebingen_bounds['max_lat']) &
            (stops_df['longitude'] >= self.tuebingen_bounds['min_lon']) &
            (stops_df['longitude'] <= self.tuebingen_bounds['max_lon']) &
            (stops_df['stop_name'].str.contains('Tübingen', case=False, na=False))
        )
        
        tuebingen_stops = stops_df[tuebing_mask].copy()
        
        logger.info(f"Filtered to {len(tuebingen_stops)} Tübingen stops")
        return tuebingen_stops
    
    def extract_tuebingen_stops(self) -> pd.DataFrame:
        """
        Extract complete Tübingen stops dataset with coordinates.
        
        Returns:
            DataFrame with Tübingen stops including coordinates
        """
        logger.info("Starting Tübingen stops extraction")
        
        # Fetch stops data
        stops_df = self.fetch_gtfs_table('stops')
        
        if stops_df.empty:
            raise ValueError("Failed to fetch stops data from GTFS API")
        
        # Filter to Tübingen
        tuebingen_stops = self.filter_tuebingen_stops(stops_df)
        
        if tuebingen_stops.empty:
            raise ValueError("No Tübingen stops found in the data")
        
        # Select and rename relevant columns
        result_df = tuebingen_stops[[
            'stop_id', 'stop_name', 'stop_geometry', 
            'longitude', 'latitude', 'municipality'
        ]].copy()
        
        # Add metadata
        result_df['data_source'] = 'GTFS_API'
        result_df['extraction_timestamp'] = pd.Timestamp.now()
        
        # Remove any remaining null coordinates
        result_df = result_df.dropna(subset=['longitude', 'latitude'])
        
        logger.info(f"Successfully extracted {len(result_df)} Tübingen stops")
        return result_df
    
    def save_stops_csv(self, stops_df: pd.DataFrame, output_path: Path):
        """Save stops data to CSV file."""
        stops_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved complete stops data to {output_path}")
    
    def save_weather_ready_csv(self, stops_df: pd.DataFrame, output_path: Path):
        """Save weather-ready CSV with minimal columns."""
        weather_df = stops_df[[
            'stop_id', 'stop_name', 'latitude', 'longitude'
        ]].copy()
        
        weather_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved weather-ready data to {output_path}")
    
    def save_coordinates_list(self, stops_df: pd.DataFrame, output_path: Path):
        """Save simple coordinates list for weather API."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Tübingen Stop Coordinates for Weather API\n")
            f.write("# Format: stop_name,latitude,longitude\n")
            f.write("# Center point for weather queries: "
                   f"{stops_df['latitude'].mean():.6f},{stops_df['longitude'].mean():.6f}\n\n")
            
            for _, stop in stops_df.iterrows():
                f.write(f"{stop['stop_name']},{stop['latitude']:.6f},{stop['longitude']:.6f}\n")
        
        logger.info(f"Saved coordinates list to {output_path}")
    
    def get_stop_statistics(self, stops_df: pd.DataFrame) -> Dict:
        """Calculate statistics for the extracted stops."""
        if stops_df.empty:
            return {}
        
        stats = {
            'total_stops': len(stops_df),
            'unique_stop_names': stops_df['stop_name'].nunique(),
            'latitude_range': {
                'min': stops_df['latitude'].min(),
                'max': stops_df['latitude'].max(),
                'center': stops_df['latitude'].mean()
            },
            'longitude_range': {
                'min': stops_df['longitude'].min(),
                'max': stops_df['longitude'].max(),
                'center': stops_df['longitude'].mean()
            },
            'municipalities': stops_df['municipality'].value_counts().to_dict()
        }
        
        return stats
