# Data Cleaning Notes

This document records data cleaning and filtering decisions made during the analysis of Tübingen bus network delays. Reference this when writing the paper's methodology section.

---

## 1. Geographic Scope

**Decision:** Focus on stops in and close to Tübingen.

**Reason:** Our data collection via the TRIAS API focused on the Tübingen area. Stops outside this region (e.g., Betzingen, Jettenburg, parts of Reutlingen) often lack GPS coordinates in our dataset.

**Impact:**

- 13 out of 23 stops for Line 7611 have no coordinates
- All Betzingen stops, Jettenburg Brunnenplatz, and some Reutlingen stops are missing
- Network graphs only display stops with valid coordinates

---

## 2. Spurious Edge Filtering

**Problem:** Some network graph edges appeared to "jump" across multiple stops, creating incorrect connections (e.g., "Tübingen Auf der Morgenstelle → Tübingen Nonnenhaus" with only 1 trip).

**Root Cause:** When intermediate stops lack coordinates, the edge generation algorithm creates edges between non-adjacent stops that happen to be consecutive in the coordinate-filtered data.

**Solution:** Filter edges using combined distance and delay rules:

- Single-trip edges (count = 1): remove if distance > 0.8 km
- Low-count edges (count 2-4): remove if distance > 1.2 km
- Low-count edges with extreme delays (count < 5, delay > 20 min): remove (likely spurious jumps)
- Higher-count edges (count ≥ 5): keep all (reliable data)

**Rationale:**

- Legitimate bus stop connections are typically < 0.8 km apart
- Multi-trip edges are more reliable
- Extreme delays with low trip counts often indicate "jumps" over missing intermediate stops
- The combination of distance AND delay catches edges that slip through distance-only filters

**Statistics (Line 5 pre-schedule example):**

- Spurious edges (count=1): median distance 2.51 km, max 3.35 km
- Valid edges (count≥2): median distance 0.39 km, max 2.61 km
- Edges like "Morgenstelle → Beethovenweg" (2 trips, 47 min delay) removed by delay rule

**Removed edges examples:**

- Tübingen Freibad → Rottenburg Bahnhof: 8.90 km (distance rule)
- Tübingen Auf der Morgenstelle → Tübingen Beethovenweg: 0.74 km, 47 min delay (delay rule)

---

## 3. Lines Without Network Data

**Decision:** Remove lines from the UI dropdown if they have insufficient data for network visualization.

**Removed lines:** 008, 101, 283, 283A, 336, 7633, E, N42, N84, N87, N90, N95, X11, X82

**Reason:** These lines have either:

- Too few records (e.g., X11 has only 114 records with 6 stops)
- No stops with coordinates in our dataset

---

## 4. Outlier Handling

**Decision:** Document but do not remove extreme delays.

**Statistics:**

- Total records: 143,213
- Extreme delays (> 30 min): 1,223 records (0.85%)
- Top disruption day: Jan 25, 2026 (253 records, lines 1, 3, 4, 6, 828)

**Rationale:**

- Extreme delays represent real service disruptions
- Using median/quantiles instead of mean mitigates outlier influence
- Removing outliers would hide genuine service issues

**Example:** Line 34 has mean delay of 17.8 min but median of only 2.0 min due to a major disruption on Dec 4, 2025 (83 min delays).

---

## 5. Schedule Change Date

**Definition:** December 14, 2025

**Usage:** Data is split into "pre" and "post" periods for comparison analysis.

---

## 6. Late Threshold

**Definition:** A bus is considered "late" if delay > 2 minutes.

**Rationale:**

- Aligns with common transit agency definitions
- Allows for minor timing variations
- More interpretable than raw delay values

---

## Summary Table

| Filter           | Threshold                  | Records Affected          | Justification                |
| ---------------- | -------------------------- | ------------------------- | ---------------------------- |
| Geographic scope | Tübingen area              | ~30% of stops lack coords | Data collection focus        |
| Spurious edges   | count=1 AND distance>1.5km | ~20 edges per line        | Coordinate gaps create jumps |
| Extreme delays   | >30 min                    | 0.85% of records          | Documented, not removed      |
| Late threshold   | >2 min                     | N/A (definition)          | Industry standard            |

---

_Last updated: January 30, 2026_
