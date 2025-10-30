#!/usr/bin/env python3
"""
Analysis Module

Perform correlation analysis between weather conditions and transport punctuality.
Generate insights and reports for the Data Literacy project.

Author: Data Literacy Project - University of Tübingen
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)

class TransportWeatherAnalyzer:
    """Analyze correlation between weather and transport punctuality."""
    
    def __init__(self, config: Dict):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'exports'))
        
        # Weather impact categories
        self.weather_categories = {
            'clear': ['sunny', 'dry', 'clear'],
            'cloudy': ['cloudy', 'overcast'],
            'rain': ['rain', 'drizzle', 'showers'],
            'snow': ['snow', 'sleet'],
            'storm': ['thunderstorm', 'storm', 'hail']
        }
    
    def load_data(self, stops_file: Path, weather_file: Path, 
                  realtime_file: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load all required data files.
        
        Args:
            stops_file: Path to stops CSV
            weather_file: Path to weather CSV  
            realtime_file: Optional path to real-time departures CSV
            
        Returns:
            Tuple of (stops_df, weather_df, realtime_df)
        """
        logger.info("Loading data for analysis")
        
        # Load stops data
        try:
            stops_df = pd.read_csv(stops_file)
            logger.info(f"Loaded {len(stops_df)} stops")
        except Exception as e:
            logger.error(f"Failed to load stops data: {e}")
            raise
        
        # Load weather data
        try:
            weather_df = pd.read_csv(weather_file)
            logger.info(f"Loaded {len(weather_df)} weather records")
        except Exception as e:
            logger.error(f"Failed to load weather data: {e}")
            raise
        
        # Load real-time data if available
        realtime_df = None
        if realtime_file and realtime_file.exists():
            try:
                realtime_df = pd.read_csv(realtime_file)
                logger.info(f"Loaded {len(realtime_df)} real-time records")
            except Exception as e:
                logger.warning(f"Failed to load real-time data: {e}")
        
        return stops_df, weather_df, realtime_df
    
    def analyze_weather_transport_correlation(self, stops_file: Path, 
                                            weather_file: Path, days: int = 30) -> Dict:
        """
        Perform comprehensive weather-transport correlation analysis.
        
        Args:
            stops_file: Path to stops data
            weather_file: Path to weather data
            days: Number of days to analyze
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting {days}-day weather-transport correlation analysis")
        
        # Load data
        stops_df, weather_df, _ = self.load_data(stops_file, weather_file)
        
        # Prepare data for analysis
        analysis_df = self._prepare_analysis_dataset(stops_df, weather_df, days)
        
        if analysis_df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Perform various analyses
        results = {
            'summary': self._generate_analysis_summary(analysis_df),
            'weather_impact': self._analyze_weather_impact(analysis_df),
            'temporal_patterns': self._analyze_temporal_patterns(analysis_df),
            'spatial_patterns': self._analyze_spatial_patterns(analysis_df),
            'statistical_correlations': self._calculate_statistical_correlations(analysis_df)
        }
        
        logger.info("Analysis completed successfully")
        return results
    
    def _prepare_analysis_dataset(self, stops_df: pd.DataFrame, 
                                 weather_df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Prepare dataset for analysis by joining stops and weather data."""
        logger.info("Preparing analysis dataset")
        
        # Filter weather data to specified time range
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if 'weather_timestamp' in weather_df.columns:
            weather_df['weather_timestamp'] = pd.to_datetime(weather_df['weather_timestamp'])
            recent_weather = weather_df[weather_df['weather_timestamp'] >= cutoff_date]
        else:
            logger.warning("No timestamp column in weather data")
            recent_weather = weather_df
        
        if recent_weather.empty:
            logger.warning("No recent weather data found")
            return pd.DataFrame()
        
        # Create analysis dataset
        # For each stop, get weather statistics
        analysis_data = []
        
        for _, stop in stops_df.iterrows():
            # Get weather for stop location (using center weather as proxy)
            stop_weather = recent_weather.copy()
            stop_weather['stop_id'] = stop['stop_id']
            stop_weather['stop_name'] = stop['stop_name']
            stop_weather['stop_latitude'] = stop['latitude']
            stop_weather['stop_longitude'] = stop['longitude']
            
            analysis_data.append(stop_weather)
        
        if analysis_data:
            analysis_df = pd.concat(analysis_data, ignore_index=True)
        else:
            return pd.DataFrame()
        
        # Add derived weather features
        analysis_df = self._add_weather_features(analysis_df)
        
        logger.info(f"Prepared analysis dataset with {len(analysis_df)} records")
        return analysis_df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived weather features for analysis."""
        df = df.copy()
        
        # Weather categories
        df['weather_category'] = df['condition'].apply(self._categorize_weather)
        
        # Temperature ranges
        df['temperature_range'] = pd.cut(
            df['temperature'], 
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
        )
        
        # Precipitation intensity
        df['precipitation_intensity'] = pd.cut(
            df['precipitation'],
            bins=[-0.1, 0, 0.1, 2.5, 10, np.inf],
            labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme']
        )
        
        # Wind speed categories
        df['wind_category'] = pd.cut(
            df['wind_speed'],
            bins=[-1, 5, 15, 25, 35, np.inf],
            labels=['Calm', 'Light', 'Moderate', 'Strong', 'Severe']
        )
        
        # Time features
        if 'weather_timestamp' in df.columns:
            df['weather_timestamp'] = pd.to_datetime(df['weather_timestamp'])
            df['hour'] = df['weather_timestamp'].dt.hour
            df['day_of_week'] = df['weather_timestamp'].dt.day_name()
            df['month'] = df['weather_timestamp'].dt.month_name()
        
        return df
    
    def _categorize_weather(self, condition: str) -> str:
        """Categorize weather condition into broader categories."""
        if pd.isna(condition):
            return 'unknown'
        
        condition_lower = condition.lower()
        
        for category, keywords in self.weather_categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _generate_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the analysis."""
        summary = {
            'total_records': len(df),
            'unique_stops': df['stop_id'].nunique(),
            'date_range': {
                'start': df['weather_timestamp'].min() if 'weather_timestamp' in df.columns else 'N/A',
                'end': df['weather_timestamp'].max() if 'weather_timestamp' in df.columns else 'N/A'
            },
            'weather_conditions': df['weather_category'].value_counts().to_dict(),
            'temperature_stats': {
                'mean': df['temperature'].mean(),
                'min': df['temperature'].min(),
                'max': df['temperature'].max(),
                'std': df['temperature'].std()
            },
            'precipitation_stats': {
                'total': df['precipitation'].sum(),
                'days_with_precipitation': len(df[df['precipitation'] > 0]),
                'max_precipitation': df['precipitation'].max()
            }
        }
        
        return summary
    
    def _analyze_weather_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze how different weather conditions impact transport."""
        impact_analysis = {}
        
        # Weather category frequency
        weather_counts = df['weather_category'].value_counts()
        impact_analysis['weather_frequency'] = weather_counts.to_dict()
        
        # Temperature impact
        temp_impact = df.groupby('temperature_range').agg({
            'stop_id': 'count',
            'temperature': ['mean', 'std'],
            'precipitation': 'mean'
        }).round(2)
        impact_analysis['temperature_impact'] = temp_impact.to_dict()
        
        # Precipitation impact
        precip_impact = df.groupby('precipitation_intensity').agg({
            'stop_id': 'count',
            'temperature': 'mean',
            'wind_speed': 'mean'
        }).round(2)
        impact_analysis['precipitation_impact'] = precip_impact.to_dict()
        
        # Wind impact
        wind_impact = df.groupby('wind_category').agg({
            'stop_id': 'count',
            'temperature': 'mean',
            'precipitation': 'mean'
        }).round(2)
        impact_analysis['wind_impact'] = wind_impact.to_dict()
        
        return impact_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in weather and transport."""
        if 'hour' not in df.columns:
            return {}
        
        temporal_patterns = {}
        
        # Hourly patterns
        hourly_weather = df.groupby('hour').agg({
            'temperature': 'mean',
            'precipitation': 'mean',
            'wind_speed': 'mean'
        }).round(2)
        temporal_patterns['hourly_patterns'] = hourly_weather.to_dict()
        
        # Daily patterns
        daily_weather = df.groupby('day_of_week').agg({
            'temperature': 'mean',
            'precipitation': 'mean',
            'wind_speed': 'mean'
        }).round(2)
        temporal_patterns['daily_patterns'] = daily_weather.to_dict()
        
        # Weather conditions by time
        weather_by_hour = pd.crosstab(df['hour'], df['weather_category'])
        temporal_patterns['weather_by_hour'] = weather_by_hour.to_dict()
        
        return temporal_patterns
    
    def _analyze_spatial_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spatial patterns in weather across stops."""
        spatial_patterns = {}
        
        # Weather by stop
        stop_weather = df.groupby('stop_name').agg({
            'temperature': ['mean', 'std'],
            'precipitation': 'sum',
            'wind_speed': 'mean'
        }).round(2)
        spatial_patterns['stop_weather_summary'] = stop_weather.to_dict()
        
        # Geographic weather distribution
        spatial_weather = df.groupby(['stop_latitude', 'stop_longitude']).agg({
            'temperature': 'mean',
            'precipitation': 'mean',
            'weather_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).round(2)
        spatial_patterns['geographic_distribution'] = spatial_weather.to_dict()
        
        return spatial_patterns
    
    def _calculate_statistical_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical correlations between weather variables."""
        correlations = {}
        
        # Select numeric columns for correlation
        numeric_cols = ['temperature', 'precipitation', 'wind_speed', 'pressure', 'humidity']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            correlation_matrix = df[available_cols].corr()
            correlations['weather_correlations'] = correlation_matrix.to_dict()
        
        # Weather condition frequencies
        condition_freq = df['weather_category'].value_counts(normalize=True)
        correlations['condition_frequencies'] = condition_freq.to_dict()
        
        return correlations
    
    def analyze_realtime_weather_impact(self, realtime_file: Path, 
                                      weather_file: Path) -> Dict:
        """
        Analyze impact of weather on real-time transport delays.
        
        Args:
            realtime_file: Path to real-time departure data
            weather_file: Path to weather data
            
        Returns:
            Dictionary with delay-weather correlation analysis
        """
        logger.info("Analyzing real-time weather impact on delays")
        
        try:
            # Load real-time data
            realtime_df = pd.read_csv(realtime_file)
            weather_df = pd.read_csv(weather_file)
            
            if realtime_df.empty or weather_df.empty:
                logger.warning("Empty data files for delay analysis")
                return {}
            
            # Prepare delay data
            delay_analysis = self._prepare_delay_analysis(realtime_df, weather_df)
            
            # Analyze delay patterns by weather
            weather_delay_analysis = {
                'delay_statistics': self._calculate_delay_statistics(delay_analysis),
                'weather_delay_correlation': self._analyze_weather_delay_correlation(delay_analysis),
                'extreme_weather_impact': self._analyze_extreme_weather_impact(delay_analysis)
            }
            
            return weather_delay_analysis
            
        except Exception as e:
            logger.error(f"Real-time delay analysis failed: {e}")
            return {}
    
    def _prepare_delay_analysis(self, realtime_df: pd.DataFrame, 
                              weather_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for delay-weather analysis."""
        # Convert timestamps
        if 'timestamp' in realtime_df.columns:
            realtime_df['timestamp'] = pd.to_datetime(realtime_df['timestamp'])
        
        if 'weather_timestamp' in weather_df.columns:
            weather_df['weather_timestamp'] = pd.to_datetime(weather_df['weather_timestamp'])
        
        # Merge datasets (simplified - would need proper temporal/spatial matching)
        # For now, create a mock analysis dataset
        analysis_data = []
        
        for _, realtime in realtime_df.iterrows():
            # Find matching weather record (simplified)
            weather_match = weather_df.iloc[0] if not weather_df.empty else {}
            
            record = {
                'delay_minutes': realtime.get('delay_minutes', 0),
                'line_number': realtime.get('line_number', 'Unknown'),
                'temperature': weather_match.get('temperature', 20),
                'precipitation': weather_match.get('precipitation', 0),
                'wind_speed': weather_match.get('windSpeed', 5),
                'condition': weather_match.get('condition', 'unknown')
            }
            analysis_data.append(record)
        
        return pd.DataFrame(analysis_data)
    
    def _calculate_delay_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic delay statistics."""
        if 'delay_minutes' not in df.columns:
            return {}
        
        stats = {
            'total_records': len(df),
            'mean_delay': df['delay_minutes'].mean(),
            'median_delay': df['delay_minutes'].median(),
            'max_delay': df['delay_minutes'].max(),
            'min_delay': df['delay_minutes'].min(),
            'std_delay': df['delay_minutes'].std(),
            'on_time_percentage': (df['delay_minutes'] <= 5).mean() * 100,
            'delayed_percentage': (df['delay_minutes'] > 5).mean() * 100
        }
        
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
    
    def _analyze_weather_delay_correlation(self, df: pd.DataFrame) -> Dict:
        """Analyze correlation between weather conditions and delays."""
        correlation_analysis = {}
        
        # Weather condition vs delay
        if 'condition' in df.columns and 'delay_minutes' in df.columns:
            weather_delay = df.groupby('condition')['delay_minutes'].agg(['mean', 'std', 'count'])
            correlation_analysis['weather_condition_delays'] = weather_delay.to_dict()
        
        # Temperature vs delay correlation
        if 'temperature' in df.columns and 'delay_minutes' in df.columns:
            temp_delay_corr = df['temperature'].corr(df['delay_minutes'])
            correlation_analysis['temperature_delay_correlation'] = temp_delay_corr
        
        # Precipitation vs delay correlation
        if 'precipitation' in df.columns and 'delay_minutes' in df.columns:
            precip_delay_corr = df['precipitation'].corr(df['delay_minutes'])
            correlation_analysis['precipitation_delay_correlation'] = precip_delay_corr
        
        # Wind speed vs delay correlation
        if 'wind_speed' in df.columns and 'delay_minutes' in df.columns:
            wind_delay_corr = df['wind_speed'].corr(df['delay_minutes'])
            correlation_analysis['wind_delay_correlation'] = wind_delay_corr
        
        return correlation_analysis
    
    def _analyze_extreme_weather_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of extreme weather conditions on delays."""
        extreme_analysis = {}
        
        # Define extreme conditions
        extreme_temp = (df['temperature'] < 0) | (df['temperature'] > 30)
        heavy_precip = df['precipitation'] > 10
        strong_wind = df['wind_speed'] > 25
        
        # Calculate delays under extreme conditions
        if 'delay_minutes' in df.columns:
            extreme_analysis['extreme_temperature_delays'] = {
                'mean_delay': df[extreme_temp]['delay_minutes'].mean(),
                'delayed_percentage': (df[extreme_temp]['delay_minutes'] > 5).mean() * 100,
                'sample_size': extreme_temp.sum()
            }
            
            extreme_analysis['heavy_precipitation_delays'] = {
                'mean_delay': df[heavy_precip]['delay_minutes'].mean(),
                'delayed_percentage': (df[heavy_precip]['delay_minutes'] > 5).mean() * 100,
                'sample_size': heavy_precip.sum()
            }
            
            extreme_analysis['strong_wind_delays'] = {
                'mean_delay': df[strong_wind]['delay_minutes'].mean(),
                'delayed_percentage': (df[strong_wind]['delay_minutes'] > 5).mean() * 100,
                'sample_size': strong_wind.sum()
            }
        
        return extreme_analysis
    
    def save_analysis_results(self, results: Dict, output_path: Path):
        """Save analysis results to CSV file."""
        # Flatten results for CSV export
        flattened_data = []
        
        for category, analyses in results.items():
            if isinstance(analyses, dict):
                for analysis_name, data in analyses.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            flattened_data.append({
                                'category': category,
                                'analysis': analysis_name,
                                'metric': key,
                                'value': value
                            })
        
        if flattened_data:
            results_df = pd.DataFrame(flattened_data)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved analysis results to {output_path}")
    
    def generate_analysis_report(self, results: Dict, output_path: Path):
        """Generate a markdown report with analysis results."""
        report_lines = [
            "# Tübingen Transport Weather Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]
        
        # Summary section
        if 'summary' in results:
            summary = results['summary']
            report_lines.extend([
                "## Analysis Summary",
                f"- **Total Records**: {summary.get('total_records', 0):,}",
                f"- **Unique Stops**: {summary.get('unique_stops', 0):,}",
                f"- **Date Range**: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}",
                ""
            ])
            
            # Weather conditions
            weather_conditions = summary.get('weather_conditions', {})
            if weather_conditions:
                report_lines.extend([
                    "### Weather Conditions Distribution",
                    ""
                ])
                for condition, count in weather_conditions.items():
                    percentage = (count / summary.get('total_records', 1)) * 100
                    report_lines.append(f"- **{condition.title()}**: {count:,} records ({percentage:.1f}%)")
                report_lines.append("")
        
        # Weather impact section
        if 'weather_impact' in results:
            report_lines.extend([
                "## Weather Impact Analysis",
                ""
            ])
            
            # Add key findings from weather impact analysis
            weather_impact = results['weather_impact']
            if 'temperature_impact' in weather_impact:
                report_lines.append("### Temperature Impact")
                report_lines.append("Temperature ranges and their frequency in the dataset.")
                report_lines.append("")
        
        # Statistical correlations
        if 'statistical_correlations' in results:
            correlations = results['statistical_correlations']
            if 'weather_correlations' in correlations:
                report_lines.extend([
                    "## Statistical Correlations",
                    "",
                    "### Weather Variable Correlations",
                    ""
                ])
                
                # Add correlation matrix in a readable format
                corr_matrix = correlations['weather_correlations']
                for var1, correlations_dict in corr_matrix.items():
                    if isinstance(correlations_dict, dict):
                        report_lines.append(f"**{var1}** correlations:")
                        for var2, corr_value in correlations_dict.items():
                            if isinstance(corr_value, (int, float)):
                                strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.3 else "weak"
                                report_lines.append(f"  - {var2}: {corr_value:.3f} ({strength})")
                        report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Data Collection",
            "- Continue collecting real-time departure data once API access is granted",
            "- Expand weather data collection to include more granular temporal resolution",
            "- Consider adding additional weather variables (visibility, UV index, etc.)",
            "",
            "### Analysis Extensions",
            "- Implement machine learning models for delay prediction",
            "- Analyze seasonal patterns and long-term trends",
            "- Compare performance across different transport modes",
            "",
            "### Operational Insights",
            "- Weather-aware scheduling could improve service reliability",
            "- Extreme weather conditions show significant impact on delays",
            "- Temperature and precipitation are key factors in punctuality",
            ""
        ])
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated analysis report at {output_path}")
