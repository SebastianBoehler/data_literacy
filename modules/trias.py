import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

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
        results = []
        for node in root.findall(".//trias:LocationResult", TRIAS_NS):
            location = node.find("trias:Location", TRIAS_NS)
            if location is None:
                continue

            stop_point_ref = location.find("trias:LocationRef/trias:StopPointRef", TRIAS_NS)
            stop_place_ref = location.find("trias:StopPlace/trias:StopPlaceRef", TRIAS_NS)
            stop_id = None
            if stop_point_ref is not None and stop_point_ref.text:
                stop_id = stop_point_ref.text
            elif stop_place_ref is not None and stop_place_ref.text:
                stop_id = stop_place_ref.text

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

            results.append(
                {
                    "stop_id": stop_id,
                    "stop_name": stop_name,
                    "latitude": float(lat_elem.text) if lat_elem is not None and lat_elem.text else None,
                    "longitude": float(lon_elem.text) if lon_elem is not None and lon_elem.text else None,
                }
            )
        stops = pd.DataFrame(results)
        return stops.dropna(subset=["stop_id"]).drop_duplicates(subset=["stop_id"])

    def fetch_departures(
        self,
        stop_id: str,
        limit: int = 20,
    ) -> pd.DataFrame:
        timestamp = _utc_now()
        payload = f"""
<StopEventRequest>
  <Location>
    <LocationRef>
      <StopPointRef>{stop_id}</StopPointRef>
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
            stop = this_call.find("trias:StopPointRef", TRIAS_NS)
            stop_name = this_call.find("trias:StopPointName/trias:Text", TRIAS_NS)
            service = this_call.find("trias:Service", TRIAS_NS)
            published_line = service.find("trias:PublishedLineName/trias:Text", TRIAS_NS) if service is not None else None
            destination = service.find("trias:DestinationText/trias:Text", TRIAS_NS) if service is not None else None
            departure = this_call.find("trias:ServiceDeparture", TRIAS_NS)
            planned = departure.find("trias:TimetabledTime", TRIAS_NS) if departure is not None else None
            estimated = departure.find("trias:EstimatedTime", TRIAS_NS) if departure is not None else None
            platform = this_call.find("trias:PlannedBay/trias:Text", TRIAS_NS)
            records.append(
                {
                    "stop_id": stop.text if stop is not None else stop_id,
                    "stop_name": stop_name.text if stop_name is not None else None,
                    "planned_time": planned.text if planned is not None else None,
                    "estimated_time": estimated.text if estimated is not None else None,
                    "line_name": published_line.text if published_line is not None else None,
                    "destination": destination.text if destination is not None else None,
                    "platform": platform.text if platform is not None else None,
                }
            )
        df = pd.DataFrame(records)
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
