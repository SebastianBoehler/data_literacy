#!/usr/bin/env python3
"""
Real-time Data Module

Handle real-time transport data from EFA-JSON and TRIAS APIs.
Provides access to departure/arrival data for delay analysis.

Author: Data Literacy Project - University of Tübingen
"""

import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RealTimeDataFetcher:
    """Fetch real-time transport data from multiple APIs."""
    
    def __init__(self, config: Dict):
        """
        Initialize real-time data fetcher.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('user_agent', 'Data-Literacy-Project-University-Tuebingen/1.0'),
            'Accept': 'application/json'
        })
        
        # API endpoints
        self.efa_base_url = config.get('efa_api_url', 'https://www.efa-bw.de/mobidata-bw')
        self.trias_endpoints = config.get('trias_endpoints', [
            'https://www.efa-bw.de/mobidata-bw/trias'
        ])
        
        # Authentication
        self.trias_requestor_ref = config.get('trias_requestor_ref')
    
    def has_trias_credentials(self) -> bool:
        """Check if TRIAS credentials are available."""
        return bool(self.trias_requestor_ref and self.trias_requestor_ref != 'YOUR_REQUESTOR_REF')
    
    def test_efa_api(self) -> Dict:
        """
        Test EFA-JSON API accessibility.
        
        Returns:
            Dictionary with test results
        """
        results = {
            'api_name': 'EFA-JSON',
            'endpoint_accessible': False,
            'stop_finder_working': False,
            'departure_monitor_working': False,
            'real_data_available': False,
            'error_message': None
        }
        
        try:
            # Test stop finder
            stop_finder_url = f"{self.efa_base_url}/XML_STOPFINDER_REQUEST"
            params = {
                'outputFormat': 'RapidJSON',
                'searchText': 'Tübingen',
                'type_sf': 'stop',
                'locationServerActive': '1'
            }
            
            logger.info("Testing EFA-JSON stop finder")
            response = self.session.get(stop_finder_url, params=params, timeout=10)
            
            if response.status_code == 200:
                results['endpoint_accessible'] = True
                results['stop_finder_working'] = True
                
                try:
                    data = response.json()
                    if 'locations' in data and len(data['locations']) > 0:
                        results['real_data_available'] = True
                        logger.info("EFA-JSON API returning real data")
                    else:
                        results['error_message'] = "API returns empty locations"
                        logger.warning("EFA-JSON API returns empty results")
                        
                except json.JSONDecodeError:
                    results['error_message'] = "Invalid JSON response"
            else:
                results['error_message'] = f"HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            results['error_message'] = str(e)
            logger.error(f"EFA-JSON API test failed: {e}")
        
        return results
    
    def test_trias_api(self) -> Dict:
        """
        Test TRIAS API accessibility.
        
        Returns:
            Dictionary with test results
        """
        results = {
            'api_name': 'TRIAS',
            'endpoint_accessible': False,
            'authentication_working': False,
            'location_search_working': False,
            'departure_data_working': False,
            'error_message': None
        }
        
        if not self.has_trias_credentials():
            results['error_message'] = "TRIAS credentials not configured"
            return results
        
        # Test with the first available endpoint
        for endpoint in self.trias_endpoints:
            try:
                logger.info(f"Testing TRIAS endpoint: {endpoint}")
                
                # Create a simple location request
                xml_request = self._create_trias_location_request("Tübingen")
                
                response = self.session.post(
                    endpoint, 
                    data=xml_request, 
                    headers={'Content-Type': 'application/xml'},
                    timeout=15
                )
                
                if response.status_code == 200:
                    results['endpoint_accessible'] = True
                    results['authentication_working'] = True
                    
                    # Try to parse response
                    try:
                        root = ET.fromstring(response.text)
                        locations = root.findall('.//{http://www.vdv.de/trias}LocationResult')
                        
                        if locations:
                            results['location_search_working'] = True
                            logger.info("TRIAS API location search working")
                        else:
                            results['error_message'] = "No locations found in response"
                            
                    except ET.ParseError as e:
                        results['error_message'] = f"XML parsing error: {e}"
                        
                elif response.status_code == 401:
                    results['error_message'] = "Authentication failed"
                elif response.status_code == 403:
                    results['error_message'] = "Access forbidden"
                else:
                    results['error_message'] = f"HTTP {response.status_code}"
                    
            except requests.exceptions.RequestException as e:
                results['error_message'] = str(e)
                logger.warning(f"TRIAS endpoint {endpoint} failed: {e}")
                continue
        
        return results
    
    def _create_trias_location_request(self, location_name: str) -> str:
        """Create TRIAS LocationInformationRequest XML."""
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        message_id = str(int(datetime.now().timestamp()))
        
        xml_request = f'''<?xml version="1.0" encoding="UTF-8"?>
<Trias version="1.2" xmlns="http://www.vdv.de/trias" xmlns:siri="http://www.siri.org.uk/siri">
    <ServiceRequest>
        <siri:RequestTimestamp>{timestamp}</siri:RequestTimestamp>
        <siri:RequestorRef>{self.trias_requestor_ref}</siri:RequestorRef>
        <siri:MessageIdentifier>{message_id}</siri:MessageIdentifier>
        <RequestPayload>
            <LocationInformationRequest>
                <InitialInput>
                    <LocationName>{location_name}</LocationName>
                </InitialInput>
                <Restrictions>
                    <Type>stop</Type>
                    <NumberOfResults>10</NumberOfResults>
                    <IncludePtModes>true</IncludePtModes>
                </Restrictions>
            </LocationInformationRequest>
        </RequestPayload>
    </ServiceRequest>
</Trias>'''
        
        return xml_request
    
    def _create_trias_stop_event_request(self, stop_point_ref: str, 
                                       location_name: str = None) -> str:
        """Create TRIAS StopEventRequest XML for departures."""
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        message_id = str(int(datetime.now().timestamp()))
        departure_time = timestamp
        
        xml_request = f'''<?xml version="1.0" encoding="UTF-8"?>
<Trias version="1.2" xmlns="http://www.vdv.de/trias" xmlns:siri="http://www.siri.org.uk/siri">
    <ServiceRequest>
        <siri:RequestTimestamp>{timestamp}</siri:RequestTimestamp>
        <siri:RequestorRef>{self.trias_requestor_ref}</siri:RequestorRef>
        <siri:MessageIdentifier>{message_id}</siri:MessageIdentifier>
        <RequestPayload>
            <StopEventRequest>
                <Location>
                    <LocationRef>
                        <StopPointRef>{stop_point_ref}</StopPointRef>'''
        
        if location_name:
            xml_request += f'''
                        <LocationName>
                            <Text>{location_name}</Text>
                        </LocationName>'''
        
        xml_request += f'''
                    </LocationRef>
                    <DepArrTime>{departure_time}</DepArrTime>
                </Location>
                <Params>
                    <NumberOfResults>20</NumberOfResults>
                    <IncludeRealtimeData>true</IncludeRealtimeData>
                    <StopEventType>departure</StopEventType>
                </Params>
            </StopEventRequest>
        </RequestPayload>
    </ServiceRequest>
</Trias>'''
        
        return xml_request
    
    def get_efa_departures(self, stop_id: str, limit: int = 20) -> List[Dict]:
        """
        Get departures using EFA-JSON API.
        
        Args:
            stop_id: EFA stop ID
            limit: Maximum number of departures
            
        Returns:
            List of departure information
        """
        logger.info(f"Fetching EFA departures for stop {stop_id}")
        
        params = {
            'outputFormat': 'RapidJSON',
            'type_dm': 'stop',
            'name_dm': stop_id,
            'useRealtime': '1',
            'limit': str(limit),
            'language': 'de'
        }
        
        try:
            url = f"{self.efa_base_url}/XML_DEPARTURE_MONITOR_REQUEST"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            departures = []
            
            if 'departureList' in data:
                for dep in data['departureList']:
                    departure = self._parse_efa_departure(dep, stop_id)
                    if departure:
                        departures.append(departure)
            
            logger.info(f"Got {len(departures)} departures from EFA-JSON")
            return departures
            
        except requests.exceptions.RequestException as e:
            logger.error(f"EFA departure request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing EFA departures: {e}")
            return []
    
    def _parse_efa_departure(self, dep_data: Dict, stop_id: str) -> Optional[Dict]:
        """Parse individual EFA departure data."""
        try:
            serving_line = dep_data.get('servingLine', {})
            date_time = dep_data.get('dateTime', {})
            real_date_time = dep_data.get('realDateTime', {})
            stop_info = dep_data.get('stopInfo', {})
            
            departure = {
                'stop_id': stop_id,
                'stop_name': stop_info.get('name', 'Unknown'),
                'line_number': serving_line.get('number', 'Unknown'),
                'line_direction': serving_line.get('direction', 'Unknown'),
                'line_type': serving_line.get('symbol', 'Unknown'),
                'operator': serving_line.get('operator', 'Unknown'),
                'departure_planned': date_time.get('time', 'Unknown'),
                'departure_realtime': real_date_time.get('time', 'Unknown'),
                'departure_date': date_time.get('date', 'Unknown'),
                'countdown': dep_data.get('countdown', 0),
                'platform': stop_info.get('platformName', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate delay
            departure['delay_minutes'] = self._calculate_delay(
                departure['departure_planned'], 
                departure['departure_realtime']
            )
            
            return departure
            
        except Exception as e:
            logger.warning(f"Error parsing EFA departure: {e}")
            return None
    
    def get_trias_departures(self, stop_point_ref: str, 
                           location_name: str = None) -> List[Dict]:
        """
        Get departures using TRIAS API.
        
        Args:
            stop_point_ref: TRIAS stop point reference
            location_name: Human readable stop name
            
        Returns:
            List of departure information
        """
        if not self.has_trias_credentials():
            logger.error("TRIAS credentials not available")
            return []
        
        logger.info(f"Fetching TRIAS departures for {stop_point_ref}")
        
        xml_request = self._create_trias_stop_event_request(
            stop_point_ref, location_name
        )
        
        for endpoint in self.trias_endpoints:
            try:
                response = self.session.post(
                    endpoint,
                    data=xml_request,
                    headers={'Content-Type': 'application/xml'},
                    timeout=15
                )
                
                if response.status_code == 200:
                    departures = self._parse_trias_departures(response.text)
                    logger.info(f"Got {len(departures)} departures from TRIAS")
                    return departures
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"TRIAS endpoint {endpoint} failed: {e}")
                continue
        
        logger.error("All TRIAS endpoints failed")
        return []
    
    def _parse_trias_departures(self, xml_response: str) -> List[Dict]:
        """Parse TRIAS departure response XML."""
        try:
            root = ET.fromstring(xml_response)
            namespaces = {
                'trias': 'http://www.vdv.de/trias',
                'siri': 'http://www.siri.org.uk/siri'
            }
            
            departures = []
            stop_events = root.findall('.//trias:StopEvent', namespaces)
            
            for event in stop_events:
                departure = self._parse_trias_stop_event(event, namespaces)
                if departure:
                    departures.append(departure)
            
            return departures
            
        except ET.ParseError as e:
            logger.error(f"TRIAS XML parsing error: {e}")
            return []
    
    def _parse_trias_stop_event(self, event_element, namespaces: Dict) -> Optional[Dict]:
        """Parse individual TRIAS stop event."""
        try:
            departure_data = {}
            
            # Extract line information
            line_name = event_element.find('.//trias:Service/trias:ServiceSection/trias:PublishedLineName/trias:Text', namespaces)
            if line_name is not None:
                departure_data['line_number'] = line_name.text
            
            # Extract destination
            destination = event_element.find('.//trias:Service/trias:DestinationText/trias:Text', namespaces)
            if destination is not None:
                departure_data['destination'] = destination.text
            
            # Extract times from ThisCall
            this_call = event_element.find('.//trias:ThisCall', namespaces)
            if this_call is not None:
                planned_time = this_call.find('.//trias:ServiceDeparture/trias:TimetabledTime', namespaces)
                estimated_time = this_call.find('.//trias:ServiceDeparture/trias:EstimatedTime', namespaces)
                
                if planned_time is not None:
                    departure_data['departure_planned'] = planned_time.text
                
                if estimated_time is not None:
                    departure_data['departure_realtime'] = estimated_time.text
                    
                    # Calculate delay
                    if planned_time is not None:
                        departure_data['delay_minutes'] = self._calculate_delay(
                            planned_time.text, estimated_time.text
                        )
            
            # Extract journey reference
            journey_ref = event_element.find('.//trias:Service/trias:JourneyRef', namespaces)
            if journey_ref is not None:
                departure_data['journey_ref'] = journey_ref.text
            
            departure_data['timestamp'] = datetime.now().isoformat()
            return departure_data
            
        except Exception as e:
            logger.warning(f"Error parsing TRIAS stop event: {e}")
            return None
    
    def _calculate_delay(self, planned_time: str, realtime_time: str) -> int:
        """Calculate delay in minutes between planned and real-time."""
        if not planned_time or not realtime_time or realtime_time == "Unknown":
            return 0
        
        try:
            # Handle different time formats
            if ':' in planned_time and len(planned_time) <= 5:
                # HH:MM format
                planned = datetime.strptime(planned_time, '%H:%M')
                realtime = datetime.strptime(realtime_time, '%H:%M')
            else:
                # ISO format
                planned = datetime.fromisoformat(planned_time.replace('Z', '+00:00'))
                realtime = datetime.fromisoformat(realtime_time.replace('Z', '+00:00'))
            
            delay = (realtime - planned).total_seconds() / 60
            return int(delay)
            
        except ValueError:
            logger.warning(f"Could not parse times: {planned_time}, {realtime_time}")
            return 0
    
    def get_haltestellenmonitor_departures(self, stop_ids: List[str], 
                                         name_dm: str = "Realtime Monitor") -> List[Dict]:
        """
        Get departures using EFA-BW Haltestellenmonitor API.
        
        Args:
            stop_ids: List of EFA stop IDs for multiple stops
            name_dm: Custom header text for the monitor
            
        Returns:
            List of departure information from multiple stops
        """
        logger.info(f"Fetching Haltestellenmonitor departures for {len(stop_ids)} stops")
        
        # Build URL parameters for multiple stops
        params = {
            'name_dm': name_dm,
            'type_dm': 'timedNode',
            'mode': 'direct',
            'sRaLP': '1',
            'locationServerActive': '1',
            'stateless': '1',
            'itdLPxx_useRealtime': 'true',
            'itdLPxx_stopname': 'true',  # Show stop names for multi-stop
            'itdLPxx_platform': 'true',
            'itdLPxx_generalInfo': 'false'
        }
        
        # Add multiple stop IDs
        for stop_id in stop_ids:
            params['stopID_dm'] = stop_id
        
        try:
            url = "https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse HTML response to extract departure data
            departures = self._parse_haltestellenmonitor_html(response.text, stop_ids, provided_stop_name=stop_name)
            
            logger.info(f"Got {len(departures)} departures from Haltestellenmonitor")
            return departures
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Haltestellenmonitor request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Haltestellenmonitor data: {e}")
            return []
    
    def get_single_stop_departures(self, stop_id: str, 
                                  name_dm: str = "Single Stop Monitor",
                                  stop_name: str = None) -> List[Dict]:
        """
        Get departures for a single stop using the proper POST request that mimics browser behavior.
        
        Args:
            stop_id: DHID stop ID (e.g., 'de:08416:10808')
            name_dm: Custom header text
            
        Returns:
            List of departure information
        """
        logger.info(f"Fetching single stop departures for {stop_id}")
        
        try:
            # First, make the initial GET request to get the session cookies
            get_params = {
                'name_dm': stop_id,
                'type_dm': 'any',
                'mode': 'direct',
                'sRaLP': '1',
                'locationServerActive': '1',
                'stateless': '1',
                'itdLPxx_useRealtime': 'true',
                'itdLPxx_stopname': 'false',
                'itdLPxx_platform': 'true',
                'itdLPxx_generalInfo': 'false'
            }
            
            get_response = self.session.get(
                "https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST", 
                params=get_params, 
                timeout=15
            )
            
            # Now make the POST request that loads the actual departure data
            post_data = {
                'language': 'de',
                'useAllStops': '1',
                'itOptionsActive': '1',
                'trITMOTvalue100': '10',
                'ptOptionsActive': '1',
                'useProxFootSearch': '0',
                'deleteAssignedStops_dm': '1',
                'itdLPxx_depOnly': '1',
                'limit': '50',  # Get more departures
                'name_dm': stop_id,
                'type_dm': 'any',
                'mode': 'direct',
                'sRaLP': '1',
                'locationServerActive': '1'
            }
            
            # Set headers to mimic browser AJAX request
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': '*/*',
                'Referer': f"https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST?name_dm={stop_id.replace(':', '%3A')}&type_dm=any&mode=direct&sRaLP=1&locationServerActive"
            }
            
            response = self.session.post(
                "https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST",
                data=post_data,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            
            # Build the monitor URL for verification
            from urllib.parse import urlencode
            monitor_url = "https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST?" + urlencode(get_params)
            
            # Parse HTML response
            departures = self._parse_haltestellenmonitor_html(response.text, [stop_id], provided_stop_name=stop_name)
            
            # Add monitor URL to each departure record for verification
            for departure in departures:
                departure['monitor_url'] = monitor_url
            
            logger.info(f"Got {len(departures)} departures for single stop")
            return departures
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Single stop departure request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing single stop data: {e}")
            return []
    
    def _parse_haltestellenmonitor_html(self, html_content: str, stop_ids: List[str], provided_stop_name: str = None) -> List[Dict]:
        """
        Parse HTML response from Haltestellenmonitor to extract departure data.
        Handles both full page HTML (GET request) and table-only HTML (POST request).
        
        Args:
            html_content: HTML response from the monitor
            stop_ids: List of stop IDs being monitored
            
        Returns:
            List of parsed departure information
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            departures = []
            
            # Extract header information for stop name (only available in full page response)
            header_text = soup.find('h1', id='headerText')
            parsed_stop_name = header_text.get_text(strip=True).replace('Abfahrten ', '') if header_text else None
            
            # Use provided stop_name if available, otherwise use parsed name, otherwise 'Unknown'
            stop_name = provided_stop_name or parsed_stop_name or 'Unknown'
            
            # Find departure table rows
            departure_rows = soup.find_all('tr', class_='departure')
            
            logger.info(f"Found {len(departure_rows)} departure rows for {stop_name}")
            
            for row in departure_rows:
                try:
                    # Extract cells from the row
                    cells = row.find_all('td')
                    logger.debug(f"Row has {len(cells)} cells: {[cell.get_text(strip=True) for cell in cells]}")
                    
                    # Handle both 7-column (POST response) and 8-column (GET response) formats
                    if len(cells) >= 6:  # Minimum columns needed
                        if len(cells) == 7:
                            # POST response format: time, icon, service, direction, stop, platform, text
                            departure_planned = self._extract_text_from_cell(cells[0])
                            line_number = self._extract_text_from_cell(cells[2])
                            direction = self._extract_text_from_cell(cells[3])
                            platform = self._extract_text_from_cell(cells[5])
                            # No real-time column in POST response, use planned time
                            departure_realtime = departure_planned
                            
                        elif len(cells) >= 8:
                            # GET response format: rbl, time, rtTime, icon, service, direction, platform, text
                            departure_planned = self._extract_text_from_cell(cells[1])
                            departure_realtime = self._extract_text_from_cell(cells[2])
                            line_number = self._extract_text_from_cell(cells[4])
                            direction = self._extract_text_from_cell(cells[5])
                            platform = self._extract_text_from_cell(cells[6])
                        else:
                            # Fallback for unexpected format
                            continue
                        
                        departure = {
                            'stop_id': stop_ids[0] if len(stop_ids) == 1 else 'multiple',
                            'stop_name': stop_name,
                            'line_number': line_number or 'Unknown',
                            'direction': direction or 'Unknown',
                            'departure_planned': departure_planned or 'Unknown',
                            'departure_realtime': departure_realtime or 'Unknown',
                            'platform': platform or 'Unknown',
                            'delay_minutes': 0,
                            'timestamp': datetime.now().isoformat(),
                            'data_source': 'haltestellenmonitor'
                        }
                        
                        # Calculate delay if real-time data available and different from planned
                        if (departure['departure_realtime'] != 'Unknown' and 
                            departure['departure_planned'] != 'Unknown' and
                            departure['departure_realtime'] != departure['departure_planned']):
                            departure['delay_minutes'] = self._calculate_delay(
                                departure['departure_planned'],
                                departure['departure_realtime']
                            )
                        
                        logger.debug(f"Parsed departure: {line_number} -> {direction}")
                        departures.append(departure)
                        
                except Exception as e:
                    logger.warning(f"Error parsing departure row: {e}")
                    continue
            
            # If no structured data found, try alternative parsing
            if not departures:
                departures = self._fallback_html_parsing(soup, stop_ids)
            
            # Log unique directions found
            if departures:
                unique_directions = list(set(dep['direction'] for dep in departures if dep['direction'] != 'Unknown'))
                logger.info(f"Found {len(departures)} departures with directions: {unique_directions}")
            
            return departures
            
        except ImportError:
            logger.error("BeautifulSoup not available for HTML parsing")
            return self._fallback_text_parsing(html_content, stop_ids)
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return []
    
    def _extract_text_from_cell(self, cell) -> str:
        """Extract clean text from HTML table cell."""
        if cell:
            # Remove any HTML tags and get clean text
            text = cell.get_text(strip=True)
            # Clean up common artifacts
            text = text.replace('\n', ' ').replace('\t', ' ').strip()
            return text
        return ''
    
    def _fallback_html_parsing(self, soup, stop_ids: List[str]) -> List[Dict]:
        """Fallback parsing method for different HTML structures."""
        departures = []
        
        # Try to find any table with departure-like content
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 4:  # Minimum columns for departure data
                    try:
                        departure = {
                            'stop_id': stop_ids[0] if len(stop_ids) == 1 else 'multiple',
                            'stop_name': self._extract_text_from_cell(cells[0]) or 'Unknown',
                            'line_number': self._extract_text_from_cell(cells[1]) or 'Unknown',
                            'direction': self._extract_text_from_cell(cells[2]) if len(cells) > 2 else 'Unknown',
                            'departure_planned': self._extract_text_from_cell(cells[3]) or 'Unknown',
                            'departure_realtime': self._extract_text_from_cell(cells[4]) if len(cells) > 4 else 'Unknown',
                            'platform': self._extract_text_from_cell(cells[5]) if len(cells) > 5 else 'Unknown',
                            'delay_minutes': 0,
                            'timestamp': datetime.now().isoformat(),
                            'data_source': 'haltestellenmonitor_fallback'
                        }
                        
                        if departure['departure_realtime'] != 'Unknown' and departure['departure_planned'] != 'Unknown':
                            departure['delay_minutes'] = self._calculate_delay(
                                departure['departure_planned'],
                                departure['departure_realtime']
                            )
                        
                        departures.append(departure)
                        
                    except Exception as e:
                        logger.warning(f"Error in fallback parsing: {e}")
                        continue
        
        return departures
    
    def _fallback_text_parsing(self, html_content: str, stop_ids: List[str]) -> List[Dict]:
        """Fallback text-based parsing when HTML parsing fails."""
        # This is a very basic fallback - would need enhancement based on actual HTML structure
        logger.warning("Using fallback text parsing - results may be limited")
        
        # Look for time patterns in the HTML
        import re
        time_pattern = r'\d{1,2}:\d{2}'
        times = re.findall(time_pattern, html_content)
        
        departures = []
        current_time = datetime.now().strftime('%H:%M')
        
        # Create mock departures based on found times (limited functionality)
        for i, time_str in enumerate(times[:10]):  # Limit to first 10 times found
            departure = {
                'stop_id': stop_ids[0] if len(stop_ids) == 1 else 'multiple',
                'stop_name': 'Parsed from HTML',
                'line_number': f'Line{i+1}',
                'direction': 'Unknown',
                'departure_planned': time_str,
                'departure_realtime': time_str,
                'platform': 'Unknown',
                'delay_minutes': 0,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'text_fallback'
            }
            departures.append(departure)
        
        return departures

    def get_api_status(self) -> Dict:
        """Get comprehensive status of all real-time APIs."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'apis': {}
        }
        
        # Test EFA-JSON
        efa_status = self.test_efa_api()
        status['apis']['efa_json'] = efa_status
        
        # Test TRIAS
        trias_status = self.test_trias_api()
        status['apis']['trias'] = trias_status
        
        # Test Haltestellenmonitor
        monitor_status = self.test_haltestellenmonitor()
        status['apis']['haltestellenmonitor'] = monitor_status
        
        # Overall assessment
        working_apis = []
        for api_name, api_status in status['apis'].items():
            if api_status.get('real_data_available') or api_status.get('location_search_working') or api_status.get('working'):
                working_apis.append(api_name)
        
        status['working_apis'] = working_apis
        status['realtime_available'] = len(working_apis) > 0
        
        return status
    
    def test_haltestellenmonitor(self) -> Dict:
        """Test Haltestellenmonitor API accessibility."""
        results = {
            'api_name': 'Haltestellenmonitor',
            'working': False,
            'single_stop_working': False,
            'multi_stop_working': False,
            'real_data_available': False,
            'error_message': None
        }
        
        try:
            # Test with a known Tübingen stop
            test_stop_id = "de:08416:10808"  # Aixer Straße
            
            logger.info("Testing Haltestellenmonitor single stop")
            single_departures = self.get_single_stop_departures(test_stop_id)
            
            if single_departures:
                results['single_stop_working'] = True
                results['real_data_available'] = True
                logger.info("Haltestellenmonitor single stop working")
            
            # Test multi-stop functionality
            test_stop_ids = ["5006221", "5000350"]  # Example Stuttgart stops
            multi_departures = self.get_haltestellenmonitor_departures(test_stop_ids)
            
            if multi_departures:
                results['multi_stop_working'] = True
                logger.info("Haltestellenmonitor multi-stop working")
            
            results['working'] = results['single_stop_working'] or results['multi_stop_working']
            
        except Exception as e:
            results['error_message'] = str(e)
            logger.error(f"Haltestellenmonitor test failed: {e}")
        
        return results
