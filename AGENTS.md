# AGENTS.md

## Overview

This document lists the **data agents (APIs, datasets, and sources)** used in our Data Literacy project at the University of Tübingen.  
The goal is to analyze **public transport punctuality** (Tübingen, Stuttgart, Filderstadt) in relation to **weather conditions**.

---

## 1. Public Transport (Realtime data)

### TRIAS 1.2 SOAP (MobiData BW)

- **Dataset Info:** https://mobidata-bw.de/dataset/trias
- **Endpoint:** `https://efa-bw.de/trias`
- **Protocol:** XML/SOAP requests compliant with TRIAS 1.2 (e.g. `LocationInformationRequest`, `StopEventRequest`).
- **What we use:**
  - `LocationInformationRequest` to discover all stop places (and their bays) within a configurable radius around Tübingen.
  - `StopEventRequest` to fetch live departures (planned vs. estimated times, line identifiers, platforms).
- **Authentication:** RequestorRef supplied by MobiData BW for academic use.
- **Purpose:** Build a snapshot of current departures per stop/bay for punctuality analysis.

---

## 2. Weather Data

### Bright Sky (DWD Observations)

- **Docs:** https://api.brightsky.dev
- **Endpoint:** `https://api.brightsky.dev/current_weather`
- **Response:** Nearest DWD station, hourly measurements (temperature, precipitation, wind, cloud cover, pressure, etc.).
- **Usage in project:** Queried per stop coordinate to enrich each departure with co-temporal weather conditions.

---

## 3. Optional / Reference Sources

Although not part of the current automated pipeline, the following resources remain relevant for future work (e.g. planned timetable baselines):

- **GTFS Static Data (MobiData BW):** https://mobidata-bw.de/dataset/soll-fahrplandaten-baden-wurttemberg
- **Historical GTFS Snapshots:** Weekly “Soll” archives (GTFS ZIP) for longitudinal analysis.
- **GeoServer Stop Geometry:** https://mobidata-bw.de/dataset/soll-fahrplandaten-baden-wurttemberg/resource/025c17df-ecab-442b-aaf0-bdb61491be57

---

## 4. Usage Summary (Current Pipeline)

| Data Type | Source & Endpoint             | Frequency | Purpose                              |
|-----------|-------------------------------|-----------|--------------------------------------|
| Realtime departures | TRIAS SOAP (`StopEventRequest`)     | ad-hoc    | Capture live delays per stop/bay     |
| Stop discovery      | TRIAS SOAP (`LocationInformationRequest`) | ad-hoc | Enumerate stops within radius        |
| Current weather     | Bright Sky (`/current_weather`)     | hourly    | Join meteorological context to stops |

---

## Notes

- All data is used **only for academic, non-commercial purposes** within the Data Literacy course.
- MobiData BW license terms and DWD Bright Sky terms apply.
- Realtime requests must respect rate limits and API terms.
