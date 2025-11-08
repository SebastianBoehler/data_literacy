import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import xml.etree.ElementTree as ET

TRIAS_BASE_URL = "https://efa-bw.de/trias"
TRIAS_HEADERS = {
    "Content-Type": "application/xml",
    "Accept": "application/xml",
    "User-Agent": "PostmanRuntime/7.39.0",
}
TRIAS_NS = {"trias": "http://www.vdv.de/trias", "siri": "http://www.siri.org.uk/siri"}


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TriasClient:
    def __init__(self, requestor_ref: str, session: Optional[requests.Session] = None) -> None:
        self.requestor_ref = requestor_ref
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.session.headers.update(TRIAS_HEADERS)

    def fetch_stops(
        self,
        center: tuple[float, float],
        radius_km: float,
        max_results: int = 200,
    ) -> pd.DataFrame:
        lat, lon = center
        radius_m = int(radius_km * 1000)
        payload = f"""
<LocationInformationRequest>
  <InitialInput>
    <GeoPosition>
      <Longitude>{lon:.6f}</Longitude>
      <Latitude>{lat:.6f}</Latitude>
    </GeoPosition>
  </InitialInput>
  <Restrictions>
    <Type>stop</Type>
    <NumberOfResults>{max_results}</NumberOfResults>
    <GeoRestriction>
      <Circle>
        <Center>
          <Longitude>{lon:.6f}</Longitude>
          <Latitude>{lat:.6f}</Latitude>
        </Center>
        <Radius>{radius_m}</Radius>
      </Circle>
    </GeoRestriction>
  </Restrictions>
</LocationInformationRequest>
"""
        root = self._execute(payload)
        seen_refs: set[str] = set()
        results: list[dict[str, object]] = []

        for node in root.findall(".//trias:LocationResult", TRIAS_NS):
            location = node.find("trias:Location", TRIAS_NS)
            if location is None:
                continue

            stop_point_elem = location.find("trias:LocationRef/trias:StopPointRef", TRIAS_NS)
            stop_place_elem = location.find("trias:StopPlace/trias:StopPlaceRef", TRIAS_NS)

            base_ref = None
            if stop_point_elem is not None and stop_point_elem.text:
                base_ref = stop_point_elem.text
            elif stop_place_elem is not None and stop_place_elem.text:
                base_ref = stop_place_elem.text

            if not base_ref:
                continue

            if base_ref in seen_refs:
                continue
            seen_refs.add(base_ref)

            primary_name = location.find("trias:LocationName/trias:Text", TRIAS_NS)
            stop_place_name = location.find("trias:StopPlace/trias:StopPlaceName/trias:Text", TRIAS_NS)
            stop_name = None
            if stop_place_name is not None and stop_place_name.text:
                stop_name = stop_place_name.text
            elif primary_name is not None and primary_name.text:
                stop_name = primary_name.text

            pos = location.find("trias:GeoPosition", TRIAS_NS)
            if pos is None:
                pos = location.find("trias:StopPlace/trias:GeoPosition", TRIAS_NS)
            lat_elem = pos.find("trias:Latitude", TRIAS_NS) if pos is not None else None
            lon_elem = pos.find("trias:Longitude", TRIAS_NS) if pos is not None else None

            probability_elem = node.find("trias:Probability", TRIAS_NS)
            probability = (
                float(probability_elem.text)
                if probability_elem is not None and probability_elem.text
                else None
            )

            results.append(
                {
                    "stop_id": base_ref,
                    "trias_ref": base_ref,
                    "stop_name": stop_name,
                    "latitude": float(lat_elem.text) if lat_elem is not None and lat_elem.text else None,
                    "longitude": float(lon_elem.text) if lon_elem is not None and lon_elem.text else None,
                    "probability": probability,
                }
            )

        stops = pd.DataFrame(results)
        if stops.empty:
            return stops
        return stops.sort_values(by="probability", ascending=False, na_position="last").reset_index(drop=True)

    def fetch_departures(
        self,
        stop_id: str,
        *,
        stop_point_ref: Optional[str] = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        timestamp = _utc_now()
        target_ref = stop_point_ref or stop_id
        payload = f"""
<StopEventRequest>
  <Location>
    <LocationRef>
      <StopPointRef>{target_ref}</StopPointRef>
    </LocationRef>
    <DepArrTime>{timestamp}</DepArrTime>
  </Location>
  <Params>
    <NumberOfResults>{limit}</NumberOfResults>
    <IncludeRealtimeData>true</IncludeRealtimeData>
    <StopEventType>departure</StopEventType>
  </Params>
</StopEventRequest>
"""
        root = self._execute(payload)
        records = []
        for node in root.findall(".//trias:StopEventResult", TRIAS_NS):
            event = node.find("trias:StopEvent", TRIAS_NS)
            if event is None:
                continue
            this_call = event.find("trias:ThisCall/trias:CallAtStop", TRIAS_NS)
            if this_call is None:
                continue
            stop_ref_elem = this_call.find("trias:StopPointRef", TRIAS_NS)
            stop_name_elem = this_call.find("trias:StopPointName/trias:Text", TRIAS_NS)
            raw_stop_ref = stop_ref_elem.text if stop_ref_elem is not None else stop_id
            stop_base_id = raw_stop_ref
            if raw_stop_ref:
                parts = raw_stop_ref.split(":")
                if len(parts) >= 3:
                    stop_base_id = ":".join(parts[:3])
            service = event.find("trias:Service", TRIAS_NS)
            service_section = service.find("trias:ServiceSection", TRIAS_NS) if service is not None else None
            published_line_elem = (
                service_section.find("trias:PublishedLineName/trias:Text", TRIAS_NS)
                if service_section is not None
                else None
            )
            route_description = service.find("trias:RouteDescription/trias:Text", TRIAS_NS) if service is not None else None
            destination_elem = service.find("trias:DestinationText/trias:Text", TRIAS_NS) if service is not None else None

            journey_ref_elem = service.find("trias:JourneyRef", TRIAS_NS) if service is not None else None
            operating_day_elem = service.find("trias:OperatingDayRef", TRIAS_NS) if service is not None else None

            line_name = None
            if published_line_elem is not None and published_line_elem.text:
                line_name = published_line_elem.text

            destination = None
            if destination_elem is not None and destination_elem.text:
                destination = destination_elem.text
            elif route_description is not None and route_description.text:
                destination = route_description.text
            departure = this_call.find("trias:ServiceDeparture", TRIAS_NS)
            planned = departure.find("trias:TimetabledTime", TRIAS_NS) if departure is not None else None
            estimated = departure.find("trias:EstimatedTime", TRIAS_NS) if departure is not None else None
            platform_elem = this_call.find("trias:PlannedBay", TRIAS_NS)
            if platform_elem is None:
                platform_elem = this_call.find("trias:ServiceDeparture/trias:Bay", TRIAS_NS)
            platform = None
            if platform_elem is not None:
                bay_text = platform_elem.find("trias:Text", TRIAS_NS)
                if bay_text is not None and bay_text.text:
                    platform = bay_text.text
                elif platform_elem.text:
                    platform = platform_elem.text
            records.append(
                {
                    "stop_id": stop_base_id,
                    "stop_point_ref": raw_stop_ref,
                    "stop_name": stop_name_elem.text if stop_name_elem is not None else None,
                    "planned_time": planned.text if planned is not None else None,
                    "estimated_time": estimated.text if estimated is not None else None,
                    "line_name": line_name,
                    "destination": destination,
                    "platform": platform,
                    "journey_ref": journey_ref_elem.text if journey_ref_elem is not None else None,
                    "operating_day_ref": operating_day_elem.text if operating_day_elem is not None else None,
                }
            )
        df = pd.DataFrame(records)
        if df.empty:
            return df
        if stop_point_ref is not None:
            df = df[df["stop_point_ref"] == stop_point_ref].reset_index(drop=True)
            if df.empty:
                return df
        berlin = "Europe/Berlin"
        df["planned_time"] = (
            pd.to_datetime(df["planned_time"], errors="coerce", utc=True)
            .dt.tz_convert(berlin)
            .dt.tz_localize(None)
        )
        df["estimated_time"] = (
            pd.to_datetime(df["estimated_time"], errors="coerce", utc=True)
            .dt.tz_convert(berlin)
            .dt.tz_localize(None)
        )
        df["delay_minutes"] = (
            (df["estimated_time"] - df["planned_time"]).dt.total_seconds() / 60
        )
        return df

    def fetch_departures_for_stop_points(
        self,
        stops: pd.DataFrame,
        limit_per_stop_point: int = 100,
    ) -> pd.DataFrame:
        if stops.empty or "stop_point_ref" not in stops.columns:
            return pd.DataFrame()

        unique = (
            stops.dropna(subset=["stop_point_ref"])
            .drop_duplicates(subset=["stop_point_ref"])
            .loc[:, ["stop_point_ref", "trias_ref", "stop_id"]]
        )

        frames: list[pd.DataFrame] = []
        for _, row in unique.iterrows():
            stop_point_ref = row["stop_point_ref"]
            base_ref = row.get("trias_ref") or row.get("stop_id")
            if pd.isna(base_ref):
                base_ref = row["stop_id"]
            frame = self.fetch_departures(
                str(base_ref), stop_point_ref=str(stop_point_ref), limit=limit_per_stop_point
            )
            if not frame.empty:
                frames.append(frame)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_trip_info(
        self,
        journey_ref: str,
        operating_day_ref: Optional[str] = None,
        *,
        include_calls: bool = True,
        include_estimated: bool = True,
        include_position: bool = True,
        include_track_sections: bool = False,
    ) -> dict[str, object]:
        params: list[str] = []
        if include_calls:
            params.append("<IncludeCalls>true</IncludeCalls>")
        if include_estimated:
            params.append("<IncludeEstimatedTimes>true</IncludeEstimatedTimes>")
        if include_position:
            params.append("<IncludePosition>true</IncludePosition>")
        if include_track_sections:
            params.append("<IncludeTrackSections>true</IncludeTrackSections>")

        params_xml = "".join(params)
        refs_xml = f"<JourneyRef>{journey_ref}</JourneyRef>"
        if operating_day_ref:
            refs_xml += f"<OperatingDayRef>{operating_day_ref}</OperatingDayRef>"

        payload = (
            "<TripInfoRequest>"
            f"{refs_xml}"
            f"<Params>{params_xml}</Params>"
            "</TripInfoRequest>"
        )

        root = self._execute(payload)
        result = root.find(".//trias:TripInfoResult", TRIAS_NS)
        if result is None:
            return {"calls": pd.DataFrame(), "service": None, "current_position": None}

        service_elem = result.find("trias:Service", TRIAS_NS)
        service_info = self._parse_service(service_elem) if service_elem is not None else None

        position_elem = result.find("trias:CurrentPosition", TRIAS_NS)
        current_position = self._parse_position(position_elem) if position_elem is not None else None

        calls_df = self._parse_calls(result)

        return {
            "calls": calls_df,
            "service": service_info,
            "current_position": current_position,
        }

    def fetch_trip_infos_for_departures(
        self,
        departures: pd.DataFrame,
        *,
        max_trips: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if departures.empty:
            return pd.DataFrame(), pd.DataFrame()

        unique = (
            departures.dropna(subset=["journey_ref"])
            .drop_duplicates(subset=["journey_ref", "operating_day_ref"])
        )

        if max_trips is not None:
            unique = unique.head(max_trips)

        call_frames: list[pd.DataFrame] = []
        position_records: list[dict[str, object]] = []

        for _, row in unique.iterrows():
            journey_ref = row["journey_ref"]
            operating_day_ref = row.get("operating_day_ref")
            trip_info = self.fetch_trip_info(journey_ref, operating_day_ref)

            service_meta = trip_info.get("service") or {}

            calls = trip_info.get("calls")
            if isinstance(calls, pd.DataFrame) and not calls.empty:
                calls = calls.copy()
                calls["journey_ref"] = service_meta.get("journey_ref") or journey_ref
                calls["operating_day_ref"] = service_meta.get("operating_day_ref") or operating_day_ref
                calls["line_name"] = service_meta.get("line_name") or row.get("line_name")
                calls["destination"] = service_meta.get("destination") or row.get("destination")
                call_frames.append(calls)

            position_meta = trip_info.get("current_position")
            if isinstance(position_meta, dict) and position_meta:
                position_record = position_meta.copy()
                position_record["journey_ref"] = journey_ref
                position_record["operating_day_ref"] = operating_day_ref
                position_record["line_name"] = service_meta.get("line_name") or row.get("line_name")
                position_record["destination"] = service_meta.get("destination") or row.get("destination")
                position_records.append(position_record)

        call_df = pd.concat(call_frames, ignore_index=True) if call_frames else pd.DataFrame()
        position_df = pd.DataFrame(position_records)
        return call_df, position_df

    def _parse_service(self, service_elem: ET.Element) -> dict[str, Optional[str]]:
        def _text(elem: Optional[ET.Element]) -> Optional[str]:
            if elem is None or elem.text is None:
                return None
            return elem.text

        journey_ref = _text(service_elem.find("trias:JourneyRef", TRIAS_NS))
        operating_day_ref = _text(service_elem.find("trias:OperatingDayRef", TRIAS_NS))

        destination = None
        destination_elem = service_elem.find("trias:DestinationText/trias:Text", TRIAS_NS)
        if destination_elem is not None and destination_elem.text:
            destination = destination_elem.text

        published_line = None
        for line_elem in service_elem.findall(
            "trias:ServiceSection/trias:PublishedLineName/trias:Text", TRIAS_NS
        ):
            if line_elem.text:
                published_line = line_elem.text
                break

        return {
            "journey_ref": journey_ref,
            "operating_day_ref": operating_day_ref,
            "destination": destination,
            "line_name": published_line,
        }

    def _parse_position(self, position_elem: ET.Element) -> dict[str, Optional[float]]:
        latitude_elem = position_elem.find("trias:GeoPosition/trias:Latitude", TRIAS_NS)
        longitude_elem = position_elem.find("trias:GeoPosition/trias:Longitude", TRIAS_NS)
        bearing_elem = position_elem.find("trias:Bearing", TRIAS_NS)

        def _float(elem: Optional[ET.Element]) -> Optional[float]:
            if elem is None or not elem.text:
                return None
            try:
                return float(elem.text)
            except ValueError:
                return None

        return {
            "latitude": _float(latitude_elem),
            "longitude": _float(longitude_elem),
            "bearing": _float(bearing_elem),
        }

    def _parse_calls(self, result_elem: ET.Element) -> pd.DataFrame:
        records: list[dict[str, object]] = []

        phase_map = {
            "PreviousCall": "previous",
            "OnwardCall": "onward",
        }

        for xml_phase, label in phase_map.items():
            for call_container in result_elem.findall(f"trias:{xml_phase}", TRIAS_NS):
                # Some TRIAS backends wrap calls in CallAtStop, others inline the elements.
                call_elem = call_container.find("trias:CallAtStop", TRIAS_NS)
                if call_elem is None:
                    call_elem = call_container
                records.append(self._call_record(call_elem, label))

        calls_df = pd.DataFrame(records)
        if calls_df.empty:
            return calls_df

        berlin = "Europe/Berlin"

        arrival_planned = pd.to_datetime(calls_df["arrival_planned"], errors="coerce", utc=True)
        arrival_estimated = pd.to_datetime(calls_df["arrival_estimated"], errors="coerce", utc=True)
        departure_planned = pd.to_datetime(calls_df["departure_planned"], errors="coerce", utc=True)
        departure_estimated = pd.to_datetime(calls_df["departure_estimated"], errors="coerce", utc=True)

        calls_df["arrival_delay_minutes"] = (
            (arrival_estimated - arrival_planned).dt.total_seconds() / 60
        )
        calls_df["departure_delay_minutes"] = (
            (departure_estimated - departure_planned).dt.total_seconds() / 60
        )

        calls_df["arrival_planned"] = (
            arrival_planned.dt.tz_convert(berlin).dt.tz_localize(None)
        )
        calls_df["arrival_estimated"] = (
            arrival_estimated.dt.tz_convert(berlin).dt.tz_localize(None)
        )
        calls_df["departure_planned"] = (
            departure_planned.dt.tz_convert(berlin).dt.tz_localize(None)
        )
        calls_df["departure_estimated"] = (
            departure_estimated.dt.tz_convert(berlin).dt.tz_localize(None)
        )

        sort_columns = ["journey_ref", "stop_sequence", "phase"]
        existing = [col for col in sort_columns if col in calls_df.columns]
        return calls_df.sort_values(by=existing).reset_index(drop=True)

    def _call_record(self, call_elem: ET.Element, phase: str) -> dict[str, object]:
        def _text(elem_path: str) -> Optional[str]:
            elem = call_elem.find(elem_path, TRIAS_NS)
            if elem is None or elem.text is None:
                return None
            return elem.text

        def _find_time(parent_tag: str) -> tuple[Optional[str], Optional[str]]:
            parent = call_elem.find(parent_tag, TRIAS_NS)
            if parent is None:
                return None, None
            planned = parent.find("trias:TimetabledTime", TRIAS_NS)
            estimated = parent.find("trias:EstimatedTime", TRIAS_NS)
            planned_text = planned.text if planned is not None else None
            estimated_text = estimated.text if estimated is not None else None
            return planned_text, estimated_text

        sequence_text = _text("trias:StopSeqNumber")
        try:
            sequence = int(sequence_text) if sequence_text is not None else None
        except ValueError:
            sequence = None

        arrival_planned, arrival_estimated = _find_time("trias:ServiceArrival")
        departure_planned, departure_estimated = _find_time("trias:ServiceDeparture")

        bay = _text("trias:PlannedBay/trias:Text") or _text("trias:ServiceDeparture/trias:Bay/trias:Text")

        return {
            "phase": phase,
            "stop_point_ref": _text("trias:StopPointRef"),
            "stop_name": _text("trias:StopPointName/trias:Text"),
            "stop_sequence": sequence,
            "platform": bay,
            "arrival_planned": arrival_planned,
            "arrival_estimated": arrival_estimated,
            "departure_planned": departure_planned,
            "departure_estimated": departure_estimated,
        }

    def fetch_departures_for_stops(
        self,
        stops: pd.DataFrame,
        limit_per_stop: int = 20,
        max_stops: Optional[int] = None,
    ) -> pd.DataFrame:
        if stops.empty:
            return pd.DataFrame()
        frames = []
        stop_ids = list(stops["stop_id"].dropna())
        if max_stops:
            stop_ids = stop_ids[:max_stops]
        for stop_id in stop_ids:
            frame = self.fetch_departures(stop_id, limit=limit_per_stop)
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _execute(self, payload: str) -> ET.Element:
        envelope = self._wrap_payload(payload)
        response = self.session.post(
            TRIAS_BASE_URL,
            data=envelope.encode("utf-8"),
            timeout=30,
        )
        response.raise_for_status()
        return ET.fromstring(response.content)

    def _wrap_payload(self, payload: str) -> str:
        timestamp = _utc_now()
        message_id = uuid.uuid4().hex
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Trias version=\"1.2\" xmlns=\"http://www.vdv.de/trias\" xmlns:siri=\"http://www.siri.org.uk/siri\">"
            "<ServiceRequest>"
            f"<siri:RequestTimestamp>{timestamp}</siri:RequestTimestamp>"
            f"<siri:RequestorRef>{self.requestor_ref}</siri:RequestorRef>"
            f"<siri:MessageIdentifier>{message_id}</siri:MessageIdentifier>"
            "<RequestPayload>"
            f"{payload}"
            "</RequestPayload>"
            "</ServiceRequest>"
            "</Trias>"
        )
