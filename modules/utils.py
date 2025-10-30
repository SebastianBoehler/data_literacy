#!/usr/bin/env python3
"""
Utilities Module

Common utility functions for the transport weather analysis project.

Author: Data Literacy Project - University of Tübingen
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'gtfs_api_url': 'https://api.mobidata-bw.de/gtfs/',
        'weather_api_url': 'https://api.brightsky.dev/',
        'efa_api_url': 'https://www.efa-bw.de/mobidata-bw',
        'trias_endpoints': [
            'https://www.efa-bw.de/mobidata-bw/trias'
        ],
        'trias_requestor_ref': 'YOUR_REQUESTOR_REF',
        'user_agent': 'Data-Literacy-Project-University-Tuebingen/1.0',
        'output_dir': 'exports',
        'delay_thresholds': {
            'minor': 5,
            'significant': 15,
            'severe': 30
        },
        'tuebingen_bounds': {
            'min_lat': 48.480,
            'max_lat': 48.570,
            'min_lon': 9.000,
            'max_lon': 9.130
        }
    }
    
    # Load from file if exists
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        except Exception as e:
            logging.warning(f"Failed to load config file {config_path}: {e}")
            logging.info("Using default configuration")
    else:
        # Create default config file
        save_config(default_config, config_path)
        logging.info(f"Created default configuration file: {config_path}")
    
    return default_config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save config file {config_path}: {e}")

def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def format_timestamp(timestamp: datetime = None) -> str:
    """
    Format timestamp for filenames and logging.
    
    Args:
        timestamp: Datetime object (defaults to now)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime('%Y%m%d_%H%M%S')

def safe_filename(filename: str) -> str:
    """
    Create safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    replacements = {
        ' ': '_',
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_'
    }
    
    safe_name = filename
    for old, new in replacements.items():
        safe_name = safe_name.replace(old, new)
    
    # Remove multiple underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinates using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    import math
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    return c * r

def find_nearest_stops(target_lat: float, target_lon: float, 
                      stops_df, max_distance_km: float = 5.0, 
                      max_stops: int = 10):
    """
    Find nearest stops to target coordinates.
    
    Args:
        target_lat, target_lon: Target coordinates
        stops_df: DataFrame with stops (must have latitude, longitude columns)
        max_distance_km: Maximum distance in kilometers
        max_stops: Maximum number of stops to return
        
    Returns:
        DataFrame with nearest stops sorted by distance
    """
    if stops_df.empty:
        return stops_df
    
    # Calculate distances
    stops_df = stops_df.copy()
    stops_df['distance_km'] = stops_df.apply(
        lambda row: calculate_distance(
            target_lat, target_lon, 
            row['latitude'], row['longitude']
        ), axis=1
    )
    
    # Filter by distance and sort
    nearest_stops = stops_df[
        stops_df['distance_km'] <= max_distance_km
    ].sort_values('distance_km').head(max_stops)
    
    return nearest_stops

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat: Latitude 
        lon: Longitude
        
    Returns:
        True if coordinates are valid
    """
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a text progress bar.
    
    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled_width = int(width * progress)
    bar = "=" * filled_width + "-" * (width - filled_width)
    
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {}
    except Exception as e:
        logging.warning(f"Failed to get memory usage: {e}")
        return {}

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        import time
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator

def validate_api_response(response_data: Dict, required_fields: list) -> bool:
    """
    Validate API response contains required fields.
    
    Args:
        response_data: API response data
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present
    """
    if not isinstance(response_data, dict):
        return False
    
    for field in required_fields:
        if field not in response_data:
            logging.warning(f"Missing required field in API response: {field}")
            return False
    
    return True

def extract_error_message(error_response: str) -> str:
    """
    Extract error message from API error response.
    
    Args:
        error_response: Raw error response string
        
    Returns:
        Extracted error message
    """
    if not error_response:
        return "Unknown error"
    
    # Try to parse as JSON
    try:
        import json
        error_data = json.loads(error_response)
        if 'error' in error_data:
            return str(error_data['error'])
        elif 'message' in error_data:
            return str(error_data['message'])
    except json.JSONDecodeError:
        pass
    
    # Try to extract from HTML error pages
    if '<title>' in error_response and '</title>' in error_response:
        start = error_response.find('<title>') + 7
        end = error_response.find('</title>')
        if start > 6 and end > start:
            return error_response[start:end].strip()
    
    # Return first line or truncated response
    lines = error_response.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        if len(first_line) < 100:
            return first_line
    
    return error_response[:100] + "..." if len(error_response) > 100 else error_response
