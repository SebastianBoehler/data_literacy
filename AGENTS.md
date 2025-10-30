# AGENTS.md

## Overview

This document lists the **data agents (APIs, datasets, and sources)** used in our Data Literacy project at the University of Tübingen.  
The goal is to analyze **public transport punctuality** (Tübingen, Stuttgart, Filderstadt) in relation to **weather conditions**.

---

## 1. Public Transport (Plan Data)

### GTFS-API (MobiData BW)

- **Docs:** https://api.mobidata-bw.de/docs/gtfs/
- **What it is:** REST API exposing GTFS tables (`stops`, `routes`, `trips`, `stop_times`, etc.) via PostgREST.
- **Use:** Retrieve schedule data and metadata (planned timetables).

### Historical GTFS Snapshots (ZIP)

- **Dataset:** https://mobidata-bw.de/dataset/soll-fahrplandaten-baden-wurttemberg
- **Contains:** Weekly “Soll” (planned) timetables for all of Baden-Württemberg since 2023.
- **Format:** GTFS ZIPs (`stops.txt`, `trips.txt`, `stop_times.txt`, `routes.txt`, etc.).
- **Example file:**  
  `https://mobidata-bw.de/gtfs-historisierung/mit_linienverlauf/2025/20251008/vvs.zip`
- **Purpose:** Use as baseline (planned times) to compute delays from realtime data.

### VVS Dataset

- **Dataset:** https://mobidata-bw.de/dataset/soll-fahrplandaten-vvs
- **Coverage:** Stuttgart region (VVS).
- **License:** CC BY 4.0

### GeoServer CSV/WFS (Stops & Platforms)

- **Resource:** https://mobidata-bw.de/dataset/soll-fahrplandaten-baden-wurttemberg/resource/025c17df-ecab-442b-aaf0-bdb61491be57
- **What it is:** Stop and platform list with coordinates and IDs.
- **Use:** Match stops to weather stations / geo analysis.

---

## 2. Public Transport (Realtime Data)

### TRIAS / EFA-JSON API

- **Dataset Info:** https://mobidata-bw.de/dataset/trias
- **Base URL:** `https://www.efa-bw.de/mobidata-bw/`
- **Format:** JSON (`outputFormat=RapidJSON`)
- **Endpoints:**
  - Stop search: `XML_STOPFINDER_REQUEST`
  - Departures: `XML_DEPARTURE_MONITOR_REQUEST`
  - Trip planner: `XML_TRIP_REQUEST2`
- **Example:**  
  https://www.efa-bw.de/mobidata-bw/XML_STOPFINDER_REQUEST?outputFormat=RapidJSON&searchText=Tuebingen

- **Access:** granted for private / academic use (approved by MobiData BW)
- **Purpose:** Get real-time estimated departures to calculate delay = estimated − timetabled.

---

## 3. Weather Data

### Bright Sky (DWD)

- **Docs:** https://api.brightsky.dev
- **Type:** REST API (no API key required)
- **What it is:** DWD hourly observation & forecast data (nearest weather station).
- **Example:**  
  https://api.brightsky.dev/weather?lat=48.52&lon=9.06&date=2025-10-20&last_date=2025-10-27

- **Use:** Join weather data (precipitation, temperature, wind) with delays.

---

## 4. Integration Platform (Reference)

### MobiData BW Developer Platform

- **URL:** https://dev-ipl.mobidata-bw.de/
- **Use:** Overview of all available open APIs (GTFS, TRIAS, GeoServer, etc.)

---

## 5. Usage Summary

| Data Type      | Source                    | Access      | Frequency | Purpose                              |
| -------------- | ------------------------- | ----------- | --------- | ------------------------------------ |
| Planned (Soll) | GTFS API / Historical ZIP | open        | weekly    | baseline schedule                    |
| Realtime (Ist) | TRIAS / EFA-JSON          | via request | live      | actual departures, delay calculation |
| Weather        | Bright Sky (DWD)          | open        | hourly    | correlate weather with delays        |
| Geo            | GeoServer WFS/CSV         | open        | static    | match stops & coordinates            |

---

## Notes

- All data is used **only for academic, non-commercial purposes** within the Data Literacy course.
- MobiData BW license terms and DWD Bright Sky terms apply.
- Realtime requests must respect rate limits and API terms.
