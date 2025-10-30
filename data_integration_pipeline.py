#!/usr/bin/env python3
"""
Data Integration Pipeline

Combines real-time transport data from EFA-BW Haltestellenmonitor 
with weather data from Bright Sky API to create a comprehensive dataset
for the Data Literacy project analyzing transport punctuality vs weather.

Author: Data Literacy Project - University of Tübingen
"""

import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import time

# Import our modules
from modules.gtfs_data import GTFSDataExtractor
from modules.realtime_data import RealTimeDataFetcher
from modules.weather_data import WeatherDataIntegration
from modules.analysis import TransportWeatherAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIntegrationPipeline:
    """Main pipeline for integrating transport and weather data."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the data integration pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', 'exports'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize modules
        self.gtfs_extractor = GTFSDataExtractor(self.config)
        self.realtime_fetcher = RealTimeDataFetcher(self.config)
        self.weather_integration = WeatherDataIntegration(self.config)
        self.analyzer = TransportWeatherAnalyzer(self.config)
        
        # Sample stops for testing (expanded to include Stuttgart for more data)
        self.sample_stops = [
            # Tübingen stops only
            {
                'stop_id': 'de:08416:10808',  # Aixer Straße, Tübingen
                'stop_name': 'Aixer Straße',
                'latitude': 48.5167,
                'longitude': 9.0583
            },
            {
                'stop_id': 'de:08416:10791',  # Tübingen Hauptbahnhof
                'stop_name': 'Tübingen Hauptbahnhof',
                'latitude': 48.5206,
                'longitude': 9.0558
            },
            {
                'stop_id': 'de:08416:10830',  # Lustnau, Tübingen
                'stop_name': 'Lustnau',
                'latitude': 48.5319,
                'longitude': 9.0697
            },
            {
                'stop_id': 'de:08416:10823',  # Derendingen, Tübingen
                'stop_name': 'Derendingen',
                'latitude': 48.5067,
                'longitude': 9.0456
            },
            {
                'stop_id': 'de:08416:10812',  # Universität, Tübingen
                'stop_name': 'Universität',
                'latitude': 48.5406,
                'longitude': 9.0583
            }
        ]
        
        logger.info("Data Integration Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'gtfs_api_url': 'https://api.mobidata-bw.de/gtfs/',
            'efa_api_url': 'https://www.efa-bw.de/mobidata-bw',
            'weather_api_url': 'https://api.brightsky.dev/',
            'user_agent': 'Data-Literacy-Project-University-Tuebingen/1.0',
            'output_dir': 'exports',
            'trias_requestor_ref': None,
            'data_collection': {
                'realtime_interval_minutes': 5,
                'weather_history_days': 7,
                'max_departures_per_stop': 20
            },
        }
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete data integration pipeline.
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info("Starting full data integration pipeline")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'datasets_created': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Step 1: Test API connectivity
            logger.info("Step 1: Testing API connectivity")
            api_status = self._test_api_connectivity()
            results['api_status'] = api_status
            results['steps_completed'].append('api_connectivity_test')
            
            # Step 2: Discover all Tübingen stops
            logger.info("Step 2: Discovering all Tübingen stops")
            all_tuebingen_stops = self._discover_tuebingen_stops()
            results['discovered_stops_count'] = len(all_tuebingen_stops)
            results['steps_completed'].append('stop_discovery')
            
            # Step 3: Collect real-time transport data for all stops
            logger.info("Step 3: Collecting real-time transport data for all stops")
            realtime_data = self._collect_realtime_transport_data(use_all_stops=True)
            results['realtime_data'] = realtime_data
            results['steps_completed'].append('realtime_transport_collection')
            
            # Step 4: Collect weather data for all stops
            logger.info("Step 4: Collecting weather data for all stops")
            weather_data = self._collect_weather_data(target_stops=all_tuebingen_stops)
            results['weather_data'] = weather_data
            results['steps_completed'].append('weather_collection')
            
            # Step 5: Integrate datasets
            logger.info("Step 5: Integrating transport and weather data")
            integrated_dataset = self._integrate_datasets(realtime_data, weather_data)
            results['integrated_dataset'] = integrated_dataset
            results['steps_completed'].append('data_integration')
            
            # Step 6: Export final dataset
            logger.info("Step 6: Exporting final dataset")
            export_results = self._export_final_dataset(integrated_dataset)
            results['export_results'] = export_results
            results['steps_completed'].append('dataset_export')
            
            # Step 7: Generate analysis summary
            logger.info("Step 7: Generating analysis summary")
            analysis_summary = self._generate_analysis_summary(integrated_dataset)
            results['analysis_summary'] = analysis_summary
            results['steps_completed'].append('analysis_summary')
            
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'success'
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'failed'
        
        return results
    
    def _test_api_connectivity(self) -> Dict:
        """Test connectivity to all required APIs."""
        logger.info("Testing API connectivity")
        
        # Test real-time APIs
        realtime_status = self.realtime_fetcher.get_api_status()
        
        # Test weather API with sample coordinates
        weather_status = {'bright_sky': {'working': False, 'error': None}}
        try:
            sample_weather = self.weather_integration.get_current_weather(48.52, 9.06)
            if sample_weather:
                weather_status['bright_sky']['working'] = True
            else:
                weather_status['bright_sky']['error'] = 'No weather data returned'
        except Exception as e:
            weather_status['bright_sky']['error'] = str(e)
        
        return {
            'realtime_apis': realtime_status,
            'weather_api': weather_status,
            'overall_status': 'working' if (
                realtime_status.get('realtime_available', False) and 
                weather_status['bright_sky']['working']
            ) else 'limited'
        }
    
    def _discover_tuebingen_stops(self, limit: int = None) -> List[Dict]:
        """
        Discover all stops containing "Tübingen" in their name using GTFS API.
        
        Args:
            limit: Optional limit on number of stops to return (for testing)
        
        Returns:
            List of stop dictionaries with id, name, and coordinates
        """
        logger.info(f"Discovering stops containing 'Tübingen' in their name (limit: {limit or 'none'})")
        
        try:
            # Fetch stops from GTFS API that contain "Tübingen"
            import requests
            
            gtfs_api_url = self.config.get('gtfs_api_url', 'https://api.mobidata-bw.de/gtfs/')
            stops_url = f"{gtfs_api_url}stops?stop_name=like.*Tübingen*"
            
            logger.info(f"Fetching stops from GTFS API: {stops_url}")
            
            response = requests.get(stops_url, timeout=30)
            response.raise_for_status()
            
            stops_data = response.json()
            
            if not stops_data:
                logger.warning("No stops data from GTFS API")
                return self.sample_stops  # Fallback to predefined stops
            
            # Convert to our format and deduplicate by stop name
            discovered_stops = {}
            for stop in stops_data:
                stop_name = stop.get('stop_name', 'Unknown')
                
                # Extract coordinates from stop_loc
                coordinates = stop.get('stop_loc', {}).get('coordinates', [0.0, 0.0])
                
                stop_info = {
                    'stop_id': stop.get('stop_id', ''),
                    'stop_name': stop_name,
                    'latitude': coordinates[1] if len(coordinates) > 1 else 0.0,  # lat is second in GeoJSON
                    'longitude': coordinates[0] if len(coordinates) > 0 else 0.0  # lon is first in GeoJSON
                }
                
                # Only keep the first occurrence (or could use other logic like most central)
                if stop_name not in discovered_stops:
                    discovered_stops[stop_name] = stop_info
            
            # Convert back to list
            discovered_stops = list(discovered_stops.values())
            
            # Apply limit if specified
            if limit and len(discovered_stops) > limit:
                discovered_stops = discovered_stops[:limit]
                logger.info(f"Limited to first {limit} stops")
            
            logger.info(f"Discovered {len(discovered_stops)} stops containing 'Tübingen'")
            
            # Save discovered stops for reference
            if discovered_stops:
                stops_df_output = pd.DataFrame(discovered_stops)
                limit_suffix = f"_limited{limit}" if limit else ""
                stops_file = self.output_dir / f"discovered_tuebingen_stops{limit_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                stops_df_output.to_csv(stops_file, index=False, encoding='utf-8')
                logger.info(f"Saved discovered stops to {stops_file}")
            
            return discovered_stops if discovered_stops else self.sample_stops
            
        except Exception as e:
            logger.error(f"Failed to discover stops: {e}")
            logger.info("Falling back to predefined stops")
            return self.sample_stops

    def _collect_realtime_transport_data(self, use_all_stops: bool = False, limit_stops: int = None) -> pd.DataFrame:
        """Collect real-time transport data by fetching from each individual stop."""
        
        # Choose which stops to use
        if use_all_stops:
            target_stops = self._discover_tuebingen_stops(limit=limit_stops)
            limit_info = f" (limited to {limit_stops})" if limit_stops else ""
            logger.info(f"Collecting real-time data for ALL discovered stops{limit_info} ({len(target_stops)} total)")
        else:
            target_stops = self.sample_stops
            logger.info(f"Collecting real-time data for predefined sample stops ({len(target_stops)} total)")
        
        all_departures = []
        
        # Fetch data for each individual stop (no more broken monitor approach)
        logger.info("Fetching individual stop data for all stops")
        
        for stop in target_stops:
            try:
                logger.info(f"Fetching departures for {stop['stop_name']} ({stop['stop_id']})")
                departures = self.realtime_fetcher.get_single_stop_departures(stop['stop_id'], stop_name=stop['stop_name'])
                
                # Extract and print the monitor URL for manual verification
                if departures:
                    monitor_url = departures[0].get('monitor_url', 'No URL available')
                    logger.info(f"🔗 Monitor URL: {monitor_url}")
                    logger.info(f"Found {len(departures)} departures for {stop['stop_name']}")
                    all_departures.extend(departures)
                else:
                    # Still try to get the URL even if no departures
                    logger.info(f"No departures available for {stop['stop_name']} at this time")
                    logger.info(f"🔗 Monitor URL for manual check: https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST?name_dm={stop['stop_id'].replace(':', '%3A')}&type_dm=any&mode=direct&itdLPxx_useRealtime=true")
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {stop['stop_name']}: {e}")
                continue
        
        if not all_departures:
            logger.warning("No real-time departure data collected - this may be outside service hours")
            return pd.DataFrame()
        
        # Convert to DataFrame
        realtime_df = pd.DataFrame(all_departures)
        
        # Save raw data
        mode = 'all' if use_all_stops else 'sample'
        limit_suffix = f"_limited{limit_stops}" if limit_stops else ""
        raw_file = self.output_dir / f"raw_realtime_data_{mode}{limit_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        realtime_df.to_csv(raw_file, index=False, encoding='utf-8')
        logger.info(f"Saved raw realtime data to {raw_file}")
        
        logger.info(f"Collected {len(realtime_df)} total departure records")
        return realtime_df
    
    def _collect_weather_data(self, target_stops: List[Dict] = None) -> pd.DataFrame:
        """Collect current weather data for all stop locations."""
        logger.info("Collecting current weather data for stop locations")
        
        all_weather_data = []
        
        # Use provided stops or fall back to sample stops
        stops_to_use = target_stops if target_stops else self.sample_stops
        
        # Get center point of all stops for weather collection
        center_lat = sum(stop['latitude'] for stop in stops_to_use) / len(stops_to_use)
        center_lon = sum(stop['longitude'] for stop in stops_to_use) / len(stops_to_use)
        
        logger.info(f"Using center coordinates: {center_lat:.4f}, {center_lon:.4f}")
        logger.info(f"Collecting weather for {len(stops_to_use)} stops")
        
        try:
            # Get current weather only
            logger.info("Fetching current weather data")
            current_weather = self.weather_integration.get_current_weather(center_lat, center_lon)
            
            if current_weather:
                logger.info(f"Got current weather data")
                
                # Add current weather data for each stop
                for stop in stops_to_use:
                    stop_weather = current_weather.copy()
                    stop_weather.update({
                        'stop_id': stop['stop_id'],
                        'stop_name': stop['stop_name'],
                        'stop_latitude': stop['latitude'],
                        'stop_longitude': stop['longitude']
                    })
                    all_weather_data.append(stop_weather)
                
                # Create DataFrame
                weather_df = pd.DataFrame(all_weather_data)
                
                # Save current weather data
                weather_file = self.output_dir / f"current_weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                weather_df.to_csv(weather_file, index=False, encoding='utf-8')
                logger.info(f"Saved current weather data to {weather_file}")
                
                return weather_df
            else:
                logger.warning("No current weather data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Weather data collection failed: {e}")
            return pd.DataFrame()
    
    def _integrate_datasets(self, realtime_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Integrate transport and weather datasets."""
        logger.info("Integrating transport and weather datasets")
        
        if realtime_df.empty or weather_df.empty:
            logger.warning("Cannot integrate - one or both datasets empty")
            return pd.DataFrame()
        
        # Remove generic monitor entries that don't represent real stops
        original_count = len(realtime_df)
        realtime_df = realtime_df[~realtime_df['stop_name'].isin([
            'Tübingen All Stops Monitor', 
            'Tübingen Multiple Stops Monitor',
            '!',
            'multiple'
        ]) & ~realtime_df['stop_id'].isin([
            'multiple'
        ])].copy()
        
        removed_count = original_count - len(realtime_df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} generic monitor entries")
        
        if realtime_df.empty:
            logger.warning("No real transport stops after filtering")
            return pd.DataFrame()
        
        # Simple integration: add current weather to each transport record
        integrated_records = []
        
        # Get the current weather record
        current_weather = weather_df.iloc[0]
        
        for _, transport_row in realtime_df.iterrows():
            # Create integrated record
            integrated_record = transport_row.to_dict()
            
            # Add current weather data
            weather_cols = ['temperature', 'precipitation', 'wind_speed', 'wind_direction', 'condition']
            for col in weather_cols:
                if col in current_weather:
                    integrated_record[col] = current_weather[col]
            
            integrated_records.append(integrated_record)
        
        if not integrated_records:
            logger.warning("No integrated records created")
            return pd.DataFrame()
        
        # Create integrated DataFrame
        integrated_df = pd.DataFrame(integrated_records)
        
        # Add collection timestamp
        integrated_df['data_collection_time'] = datetime.now().isoformat()
        
        logger.info(f"Created integrated dataset with {len(integrated_df)} records and {len(integrated_df.columns)} columns")
        
        return integrated_df
    
    def _export_final_dataset(self, integrated_df: pd.DataFrame) -> Dict:
        """Export the final integrated dataset."""
        logger.info("Exporting final dataset")
        
        if integrated_df.empty:
            logger.warning("No data to export")
            return {'status': 'failed', 'reason': 'Empty dataset'}
        
        # Generate timestamp for files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        export_results = {
            'status': 'success',
            'files_created': [],
            'record_count': len(integrated_df)
        }
        
        try:
            # Export main dataset
            main_file = self.output_dir / f"transport_weather_integrated_{timestamp}.csv"
            integrated_df.to_csv(main_file, index=False, encoding='utf-8')
            export_results['files_created'].append(str(main_file))
            
            logger.info(f"Exported {len(export_results['files_created'])} files")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            export_results['status'] = 'failed'
            export_results['error'] = str(e)
        
        return export_results
    
    def _calculate_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the dataset."""
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_records': len(df),
            'unique_stops': df['stop_id'].nunique() if 'stop_id' in df.columns else 0,
            'unique_lines': df['line_number'].nunique() if 'line_number' in df.columns else 0,
            'columns_available': list(df.columns),
            'date_range': {
                'start': df['transport_timestamp'].min() if 'transport_timestamp' in df.columns else 'N/A',
                'end': df['transport_timestamp'].max() if 'transport_timestamp' in df.columns else 'N/A'
            }
        }
        
        # Delay statistics (only if delay data available)
        if 'delay_minutes' in df.columns:
            summary['delay_statistics'] = {
                'mean_delay': df['delay_minutes'].mean(),
                'median_delay': df['delay_minutes'].median(),
                'max_delay': df['delay_minutes'].max(),
                'on_time_percentage': (df['delay_minutes'] == 0).mean() * 100,
                'delayed_percentage': (df['delay_minutes'] > 5).mean() * 100
            }
        
        # Weather statistics (only if weather data available)
        if 'temperature' in df.columns:
            summary['weather_statistics'] = {
                'mean_temperature': df['temperature'].mean(),
                'total_precipitation': df['precipitation'].sum() if 'precipitation' in df.columns else 0,
                'mean_wind_speed': df['wind_speed'].mean() if 'wind_speed' in df.columns else 0,
                'weather_conditions': df['condition'].value_counts().to_dict() if 'condition' in df.columns else {}
            }
            
            # Correlation analysis (only if both delay and weather data available)
            if 'delay_minutes' in df.columns:
                summary['correlation_summary'] = {
                    'delay_temperature_corr': df['delay_minutes'].corr(df['temperature']) if 'temperature' in df.columns else 'N/A',
                    'delay_precipitation_corr': df['delay_minutes'].corr(df['precipitation']) if 'precipitation' in df.columns else 'N/A',
                    'delay_wind_corr': df['delay_minutes'].corr(df['wind_speed']) if 'wind_speed' in df.columns else 'N/A'
                }
        
        return summary
    
    def _generate_analysis_summary(self, integrated_df: pd.DataFrame) -> Dict:
        """Generate a brief analysis summary."""
        if integrated_df.empty:
            return {'status': 'no_data'}
        
        summary = {
            'key_findings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Key findings
        mean_delay = integrated_df['delay_minutes'].mean()
        on_time_rate = (integrated_df['delay_minutes'] == 0).mean() * 100
        
        summary['key_findings'] = [
            f"Average delay: {mean_delay:.1f} minutes",
            f"On-time performance: {on_time_rate:.1f}%",
            f"Total departures analyzed: {len(integrated_df)}",
            f"Weather conditions during data collection: {integrated_df['condition'].value_counts().index[0] if not integrated_df['condition'].empty else 'Unknown'}"
        ]
        
        # Data quality assessment
        summary['data_quality'] = {
            'completeness': {
                'transport_data': len(integrated_df),
                'weather_data': len(integrated_df.dropna(subset=['temperature'])),
                'missing_weather_records': len(integrated_df) - len(integrated_df.dropna(subset=['temperature']))
            },
            'timeliness': {
                'data_collection_time': datetime.now().isoformat(),
                'freshest_departure': integrated_df['transport_timestamp'].max() if not integrated_df.empty else None
            }
        }
        
        # Recommendations
        summary['recommendations'] = [
            "Continue regular data collection to build time series",
            "Expand to more stops for better geographic coverage",
            "Add historical weather data for trend analysis",
            "Implement delay prediction models using weather features"
        ]
        
        return summary
    
    def run_sample_collection(self, use_all_stops: bool = False, limit_stops: int = None) -> Dict:
        """Run sample data collection using only real data."""
        logger.info(f"Running sample data collection (all stops: {use_all_stops}, limit: {limit_stops or 'none'})")
        
        try:
            # Test APIs
            api_status = self._test_api_connectivity()
            
            # Get target stops
            if use_all_stops:
                target_stops = self._discover_tuebingen_stops(limit=limit_stops)
                logger.info(f"Using {len(target_stops)} discovered Tübingen stops")
            else:
                target_stops = self.sample_stops
                logger.info(f"Using {len(target_stops)} sample stops")
            
            # Collect real data only
            realtime_data = self._collect_realtime_transport_data(use_all_stops=use_all_stops, limit_stops=limit_stops)
            weather_data = self._collect_weather_data(target_stops=target_stops)
            
            # Create integrated dataset with real data only
            if not realtime_data.empty and not weather_data.empty:
                integrated_data = self._integrate_datasets(realtime_data, weather_data)
                
                # Export with appropriate naming
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                mode = 'all' if use_all_stops else 'sample'
                limit_suffix = f"_limited{limit_stops}" if limit_stops else ""
                filename = f"{mode}_dataset{limit_suffix}_{timestamp}.csv"
                output_file = self.output_dir / filename
                integrated_data.to_csv(output_file, index=False, encoding='utf-8')
                
                return {
                    'status': 'success',
                    'mode': f"{mode}_stops" + (f"_limited{limit_stops}" if limit_stops else ""),
                    'api_status': api_status,
                    'records_collected': len(integrated_data),
                    'stops_processed': len(target_stops),
                    'output_file': str(output_file),
                    'summary': self._calculate_dataset_summary(integrated_data)
                }
            else:
                return {
                    'status': 'limited_data',
                    'reason': 'Insufficient real data collected (may be outside service hours)',
                    'api_status': api_status,
                    'stops_processed': len(target_stops),
                    'realtime_records': len(realtime_data) if not realtime_data.empty else 0,
                    'weather_records': len(weather_data) if not weather_data.empty else 0
                }
                
        except Exception as e:
            logger.error(f"Sample collection failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Main function to run the data integration pipeline."""
    print("🚀 Data Literacy Project - Transport Weather Integration Pipeline")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataIntegrationPipeline()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Transport Weather Data Integration Pipeline')
    parser.add_argument('--sample', action='store_true', help='Run sample data collection with predefined stops only')
    parser.add_argument('--all-stops', action='store_true', help='Use all discovered Tübingen stops (default behavior)')
    parser.add_argument('--limit', type=int, help='Limit number of stops to process (for testing)')
    parser.add_argument('--full', action='store_true', help='Run full pipeline (future feature)')
    args = parser.parse_args()
    
    if args.sample:
        print("📊 Running sample data collection with predefined stops...")
        result = pipeline.run_sample_collection(use_all_stops=False)
    elif args.full:
        print("🔄 Running full pipeline with all Tübingen stops...")
        result = pipeline.run_full_pipeline()
    else:
        # Default behavior: use all discovered stops with optional limit
        limit_info = f" (limited to {args.limit})" if args.limit else ""
        print(f"🔄 Running pipeline with all discovered Tübingen stops{limit_info}...")
        result = pipeline.run_sample_collection(use_all_stops=True, limit_stops=args.limit)
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 RESULTS SUMMARY")
    print("=" * 70)
    
    if result['status'] == 'success':
        print(f"✅ Status: {result['status'].upper()}")
        print(f"📊 Records collected: {result['records_collected']}")
        
        if 'stops_processed' in result:
            print(f"🚏 Stops processed: {result['stops_processed']}")
        
        if 'output_file' in result:
            print(f"💾 Output file: {result['output_file']}")
        
        if 'summary' in result:
            summary = result['summary']
            if 'delay_statistics' in summary:
                print(f"⏱️  Average delay: {summary['delay_statistics']['mean_delay']:.1f} minutes")
                print(f"🎯 On-time rate: {summary['delay_statistics']['on_time_percentage']:.1f}%")
            
            if 'weather_statistics' in summary:
                print(f"🌡️  Average temperature: {summary['weather_statistics']['mean_temperature']:.1f}°C")
                print(f"🌧️  Total precipitation: {summary['weather_statistics']['total_precipitation']:.1f}mm")
        
        if 'mode' in result:
            print(f"🔍 Mode: {result['mode']}")
        
        if 'discovered_stops_count' in result:
            print(f"🔍 Discovered stops: {result['discovered_stops_count']}")
        
        if 'note' in result:
            print(f"📝 Note: {result['note']}")
            
    else:
        print(f"❌ Status: {result['status'].upper()}")
        print(f"🔍 Error: {result.get('error', result.get('reason', 'Unknown error'))}")
    
    print("\n🏁 Pipeline execution completed!")

if __name__ == "__main__":
    main()
