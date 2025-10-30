#!/usr/bin/env python3
"""
Tübingen Transport Weather Analysis - Main Entry Point

Modular analysis system for studying the relationship between weather conditions
and public transport punctuality in Tübingen.

Author: Data Literacy Project - University of Tübingen
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from gtfs_data import GTFSDataExtractor
from weather_data import WeatherDataIntegration
from realtime_data import RealTimeDataFetcher
from analysis import TransportWeatherAnalyzer
from utils import setup_logging, load_config

def main():
    """Main entry point for the transport weather analysis system."""
    parser = argparse.ArgumentParser(
        description="Tübingen Transport Weather Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py extract-stops                    # Extract all Tübingen stops
  python main.py weather-integration              # Integrate weather data
  python main.py realtime-test                    # Test real-time API access
  python main.py analysis --days 30               # Run 30-day analysis
  python main.py full-analysis                    # Complete analysis pipeline
        """
    )
    
    parser.add_argument(
        "action",
        choices=[
            "extract-stops",
            "weather-integration", 
            "realtime-test",
            "analysis",
            "full-analysis"
        ],
        help="Analysis action to perform"
    )
    
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days for analysis (default: 7)"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="exports",
        help="Output directory for results (default: exports)"
    )
    
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=== TÜBINGEN TRANSPORT WEATHER ANALYSIS ===")
    print(f"Action: {args.action}")
    print(f"Output Directory: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.action == "extract-stops":
            extract_tuebingen_stops(config, output_dir)
            
        elif args.action == "weather-integration":
            integrate_weather_data(config, output_dir)
            
        elif args.action == "realtime-test":
            test_realtime_access(config)
            
        elif args.action == "analysis":
            run_analysis(config, output_dir, args.days)
            
        elif args.action == "full-analysis":
            run_full_analysis(config, output_dir, args.days)
            
        print(f"\n✅ {args.action} completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in {args.action}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def extract_tuebingen_stops(config: dict, output_dir: Path):
    """Extract all Tübingen transport stops with coordinates."""
    print("\n--- EXTRACTING TÜBINGEN STOPS ---")
    
    extractor = GTFSDataExtractor(config)
    stops_df = extractor.extract_tuebingen_stops()
    
    # Save results
    stops_file = output_dir / "tuebingen_stops_coordinates.csv"
    weather_file = output_dir / "tuebingen_stops_for_weather.csv"
    coord_file = output_dir / "tuebingen_coordinates_list.txt"
    
    extractor.save_stops_csv(stops_df, stops_file)
    extractor.save_weather_ready_csv(stops_df, weather_file)
    extractor.save_coordinates_list(stops_df, coord_file)
    
    print(f"✅ Extracted {len(stops_df)} stops")
    print(f"📁 Files saved to {output_dir}")

def integrate_weather_data(config: dict, output_dir: Path):
    """Integrate weather data with transport stops."""
    print("\n--- INTEGRATING WEATHER DATA ---")
    
    weather = WeatherDataIntegration(config)
    
    # Load stops data
    stops_file = output_dir / "tuebingen_stops_for_weather.csv"
    if not stops_file.exists():
        raise FileNotFoundError(f"Stops file not found: {stops_file}")
    
    stops_df = weather.load_stops_data(stops_file)
    
    # Get weather data for Tübingen center
    center_lat = stops_df['latitude'].mean()
    center_lon = stops_df['longitude'].mean()
    
    weather_data = weather.get_historical_weather(
        center_lat, center_lon, days=30
    )
    
    # Create integrated dataset
    integrated_df = weather.create_integrated_dataset(
        stops_df, weather_data
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    integrated_file = output_dir / f"tuebingen_weather_transport_{timestamp}.csv"
    
    weather.save_integrated_data(integrated_df, integrated_file)
    
    print(f"✅ Integrated weather data for {len(integrated_df)} records")
    print(f"📁 Saved to {integrated_file}")

def test_realtime_access(config: dict):
    """Test real-time data API access."""
    print("\n--- TESTING REAL-TIME API ACCESS ---")
    
    realtime = RealTimeDataFetcher(config)
    
    # Test EFA-JSON API
    print("Testing EFA-JSON API...")
    efa_results = realtime.test_efa_api()
    
    # Test TRIAS API (if credentials available)
    if realtime.has_trias_credentials():
        print("Testing TRIAS API...")
        trias_results = realtime.test_trias_api()
    else:
        print("⚠️  TRIAS API credentials not available")
        print("📧 Request access from: mobidata-bw@nvbw.de")
    
    print("✅ Real-time API testing completed")

def run_analysis(config: dict, output_dir: Path, days: int):
    """Run transport weather analysis."""
    print(f"\n--- RUNNING {days}-DAY ANALYSIS ---")
    
    analyzer = TransportWeatherAnalyzer(config)
    
    # Load data
    stops_file = output_dir / "tuebingen_stops_coordinates.csv"
    weather_file = output_dir / "tuebingen_stops_for_weather.csv"
    
    if not stops_file.exists() or not weather_file.exists():
        raise FileNotFoundError("Required data files not found. Run extract-stops first.")
    
    # Perform analysis
    results = analyzer.analyze_weather_transport_correlation(
        stops_file, weather_file, days=days
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"analysis_results_{timestamp}.csv"
    report_file = output_dir / f"analysis_report_{timestamp}.md"
    
    analyzer.save_analysis_results(results, results_file)
    analyzer.generate_analysis_report(results, report_file)
    
    print(f"✅ Analysis completed for {days} days")
    print(f"📁 Results saved to {results_file}")
    print(f"📄 Report saved to {report_file}")

def run_full_analysis(config: dict, output_dir: Path, days: int):
    """Run complete analysis pipeline."""
    print("\n--- RUNNING FULL ANALYSIS PIPELINE ---")
    
    # Step 1: Extract stops
    extract_tuebingen_stops(config, output_dir)
    
    # Step 2: Integrate weather data  
    integrate_weather_data(config, output_dir)
    
    # Step 3: Test real-time access
    test_realtime_access(config)
    
    # Step 4: Run analysis
    run_analysis(config, output_dir, days)
    
    print("\n🎉 FULL ANALYSIS PIPELINE COMPLETED!")

if __name__ == "__main__":
    main()
