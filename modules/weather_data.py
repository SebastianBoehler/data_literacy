#!/usr/bin/env python3
"""
Weather Data Module

Integrate weather data from Bright Sky API with transport stops.
Provides historical and current weather data for correlation analysis.

Author: Data Literacy Project - University of Tübingen
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WeatherDataIntegration:
    """Integrate weather data with transport stops for analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize weather data integration.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config
        self.base_url = config.get('weather_api_url', 'https://api.brightsky.dev/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('user_agent', 'Data-Literacy-Project-University-Tuebingen/1.0')
        })
    
    def get_weather_data(self, lat: float, lon: float, 
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch weather data for specific coordinates and date range.
        
        Args:
            lat: Latitude
            lon: Longitude  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with weather data
        """
        # Default to last 30 days if no dates provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}weather"
        params = {
            'lat': lat,
            'lon': lon,
            'date': start_date,
            'last_date': end_date
        }
        
        try:
            logger.info(f"Fetching weather data for {lat}, {lon} from {start_date} to {end_date}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'weather' not in data:
                logger.warning("No weather data in response")
                return pd.DataFrame()
            
            weather_df = pd.DataFrame(data['weather'])
            logger.info(f"Loaded {len(weather_df)} weather records")
            
            return weather_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return pd.DataFrame()
    
    def get_historical_weather(self, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """
        Get historical weather data for specified number of days.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical weather data
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.get_weather_data(lat, lon, start_date, end_date)
    
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """
        Get current weather conditions for coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with current weather data
        """
        url = f"{self.base_url}weather"
        params = {
            'lat': lat,
            'lon': lon,
            'date': datetime.now().strftime('%Y-%m-%d')  # Add required date parameter
        }
        
        try:
            logger.info(f"Fetching current weather for {lat}, {lon}")
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'weather' in data and len(data['weather']) > 0:
                current = data['weather'][0]
                return {
                    'timestamp': current.get('timestamp'),
                    'temperature': current.get('temperature'),
                    'precipitation': current.get('precipitation'),
                    'wind_speed': current.get('windSpeed'),
                    'wind_direction': current.get('windDirection'),
                    'condition': current.get('condition'),
                    'icon': current.get('icon')
                }
            else:
                logger.warning("No current weather data available")
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current weather: {e}")
            return {}
    
    def load_stops_data(self, stops_file: Path) -> pd.DataFrame:
        """
        Load stops data from CSV file.
        
        Args:
            stops_file: Path to stops CSV file
            
        Returns:
            DataFrame with stops data
        """
        try:
            logger.info(f"Loading stops data from {stops_file}")
            df = pd.read_csv(stops_file)
            logger.info(f"Loaded {len(df)} stops")
            return df
        except Exception as e:
            logger.error(f"Failed to load stops data: {e}")
            return pd.DataFrame()
    
    def create_weather_grid(self, stops_df: pd.DataFrame, grid_size: int = 3) -> List[Tuple[float, float]]:
        """
        Create a grid of coordinate points for weather data collection.
        
        Args:
            stops_df: DataFrame with stops coordinates
            grid_size: Number of grid points in each dimension
            
        Returns:
            List of (latitude, longitude) tuples
        """
        if stops_df.empty:
            return []
        
        lat_min, lat_max = stops_df['latitude'].min(), stops_df['latitude'].max()
        lon_min, lon_max = stops_df['longitude'].min(), stops_df['longitude'].max()
        
        # Create grid points
        lat_step = (lat_max - lat_min) / (grid_size - 1)
        lon_step = (lon_max - lon_min) / (grid_size - 1)
        
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat = lat_min + i * lat_step
                lon = lon_min + j * lon_step
                grid_points.append((lat, lon))
        
        logger.info(f"Created {len(grid_points)} grid points for weather data")
        return grid_points
    
    def collect_weather_for_grid(self, grid_points: List[Tuple[float, float]], 
                                days: int = 7) -> Dict[Tuple[float, float], pd.DataFrame]:
        """
        Collect weather data for all grid points.
        
        Args:
            grid_points: List of (latitude, longitude) tuples
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping coordinates to weather DataFrames
        """
        weather_data = {}
        
        for i, (lat, lon) in enumerate(grid_points):
            logger.info(f"Fetching weather for grid point {i+1}/{len(grid_points)}: {lat}, {lon}")
            
            weather_df = self.get_historical_weather(lat, lon, days)
            if not weather_df.empty:
                weather_data[(lat, lon)] = weather_df
        
        logger.info(f"Collected weather data for {len(weather_data)} grid points")
        return weather_data
    
    def create_integrated_dataset(self, stops_df: pd.DataFrame, 
                                weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create integrated dataset combining stops and weather data.
        
        Args:
            stops_df: DataFrame with stops data
            weather_df: DataFrame with weather data
            
        Returns:
            Integrated DataFrame with stops and weather information
        """
        if stops_df.empty or weather_df.empty:
            logger.warning("Cannot create integrated dataset - empty input data")
            return pd.DataFrame()
        
        # Get weather statistics for each stop location
        integrated_data = []
        
        for _, stop in stops_df.iterrows():
            stop_lat = stop['latitude']
            stop_lon = stop['longitude']
            
            # Get weather for nearest point (simplified - using center weather)
            center_weather = weather_df.copy()
            
            # Add stop information to each weather record
            for _, weather_record in center_weather.iterrows():
                integrated_record = {
                    'stop_id': stop['stop_id'],
                    'stop_name': stop['stop_name'],
                    'stop_latitude': stop_lat,
                    'stop_longitude': stop_lon,
                    'weather_timestamp': weather_record.get('timestamp'),
                    'temperature': weather_record.get('temperature'),
                    'precipitation': weather_record.get('precipitation'),
                    'wind_speed': weather_record.get('windSpeed'),
                    'wind_direction': weather_record.get('windDirection'),
                    'condition': weather_record.get('condition'),
                    'pressure': weather_record.get('pressure'),
                    'humidity': weather_record.get('relativeHumidity')
                }
                integrated_data.append(integrated_record)
        
        integrated_df = pd.DataFrame(integrated_data)
        logger.info(f"Created integrated dataset with {len(integrated_df)} records")
        
        return integrated_df
    
    def calculate_weather_statistics(self, weather_df: pd.DataFrame) -> Dict:
        """
        Calculate weather statistics for analysis.
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Dictionary with weather statistics
        """
        if weather_df.empty:
            return {}
        
        stats = {
            'total_records': len(weather_df),
            'date_range': {
                'start': weather_df['timestamp'].min(),
                'end': weather_df['timestamp'].max()
            },
            'temperature': {
                'mean': weather_df['temperature'].mean(),
                'min': weather_df['temperature'].min(),
                'max': weather_df['temperature'].max(),
                'std': weather_df['temperature'].std()
            },
            'precipitation': {
                'total': weather_df['precipitation'].sum(),
                'days_with_rain': len(weather_df[weather_df['precipitation'] > 0]),
                'max_daily': weather_df['precipitation'].max()
            },
            'wind_speed': {
                'mean': weather_df['windSpeed'].mean(),
                'max': weather_df['windSpeed'].max()
            },
            'conditions': weather_df['condition'].value_counts().to_dict()
        }
        
        return stats
    
    def save_integrated_data(self, integrated_df: pd.DataFrame, output_path: Path):
        """Save integrated dataset to CSV file."""
        integrated_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved integrated data to {output_path}")
    
    def save_weather_summary(self, weather_stats: Dict, output_path: Path):
        """Save weather statistics summary to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Weather Data Summary\n\n")
            f.write(f"Total Records: {weather_stats.get('total_records', 0)}\n")
            f.write(f"Date Range: {weather_stats.get('date_range', {}).get('start', 'N/A')} to ")
            f.write(f"{weather_stats.get('date_range', {}).get('end', 'N/A')}\n\n")
            
            temp_stats = weather_stats.get('temperature', {})
            f.write("Temperature Statistics:\n")
            f.write(f"  Mean: {temp_stats.get('mean', 0):.1f}°C\n")
            f.write(f"  Min: {temp_stats.get('min', 0):.1f}°C\n")
            f.write(f"  Max: {temp_stats.get('max', 0):.1f}°C\n")
            f.write(f"  Std Dev: {temp_stats.get('std', 0):.1f}°C\n\n")
            
            precip_stats = weather_stats.get('precipitation', {})
            f.write("Precipitation Statistics:\n")
            f.write(f"  Total: {precip_stats.get('total', 0):.1f}mm\n")
            f.write(f"  Days with Rain: {precip_stats.get('days_with_rain', 0)}\n")
            f.write(f"  Max Daily: {precip_stats.get('max_daily', 0):.1f}mm\n")
        
        logger.info(f"Saved weather summary to {output_path}")
