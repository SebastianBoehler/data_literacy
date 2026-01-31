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

**Solution:** Filter edges using distance-based rules:

- Single-trip edges (count = 1): remove if distance > 1.0 km
- Low-count edges (count 2-3): remove if distance > 1.5 km
- Higher-count edges (count ≥ 4): keep all (reliable data)

**Rationale:**

- Legitimate bus stop connections are typically < 1 km apart
- Multi-trip edges are more reliable
- We don't filter by delay since some lines (e.g., Line 7625) genuinely have high delays
- Some imperfect edges may remain, but this is preferable to over-filtering

**Trade-offs:**

- Lines with sparse data (few trips) may still have some spurious edges
- Lines with genuinely high delays are preserved (previous delay-based filter removed them)
- Direction info (H/R in journey_ref) could help but adds complexity

**Bug fix:** Edge count now uses `size()` instead of `count()` to properly count rows with NaN delay values. Previously, edges where all records had NaN delays showed count=0.

**Removed edges examples:**

- Tübingen Freibad → Rottenburg Bahnhof: 8.90 km (distance rule)
- Tübingen Hauptbahnhof Süd → Tübingen Auf der Morgenstelle: 3.35 km (distance rule)

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

## 7. Line-Specific Data Coverage Issues

**Finding:** Some lines have sparse stop-sequence coverage due to TRIAS API snapshot timing.

**Example - Line 21:**

- Total records: 1,144 across 106 journeys
- Median stops per journey: **4** (vs 30+ for well-covered lines)
- 50% of records at stop_sequence=1 (first stop only)
- This is a **data collection artifact**, not a real schedule issue

**Well-covered lines for comparison:**
| Line | Records | Journeys | Avg Stops/Journey | % at stop_seq=1 |
|------|---------|----------|-------------------|-----------------|
| 5 | 33,602 | 1,061 | 31.7 | 10.4% |
| 3 | 22,767 | 789 | 28.9 | 9.9% |
| 1 | 19,053 | 554 | 34.4 | 10.6% |
| 21 | 1,144 | 106 | 10.8 | **50.1%** |

**Impact on analysis:**

- Network graphs for sparse lines show incomplete routes
- Edge counts are unreliable for lines with low stop coverage
- Pre/post schedule change comparisons should use well-covered lines

**Recommendation:** For network visualizations comparing pre/post periods, use lines with:

- At least 50 journeys in both periods
- Average 15+ stops per journey
- Consistent stop coverage across periods

---

## 8. Schedule Change Edge Artifacts

**Finding:** Combining pre and post schedule change data creates spurious edges when routes changed.

**Root Cause:** The December 14, 2025 schedule change altered some bus routes. When building network edges from consecutive stops (using stop_sequence), the same stop_sequence number may refer to different stops before vs after the change.

**Example - Line 5:**

- Pre-change: stop_seq 2 = "Neckarbrücke", stop_seq 3 = "Wilhelmstraße"
- Post-change: stop_seq 2 = "Danziger Straße", stop_seq 3 = "Derend. Käppele"
- Combining both creates spurious edge: "Derend. Käppele → Wilhelmstraße"

**Solution:** Process pre and post schedule change data **separately** when building edges, then merge the edge statistics. This ensures edges are only created between stops that are actually consecutive within the same schedule period.

**Implementation:**

```python
# Split by schedule change
edges_pre = build_edges_for_period(data_pre)
edges_post = build_edges_for_period(data_post)
# Merge edge statistics
edges = pd.concat([edges_pre, edges_post])
```

**Impact:** Reduces spurious long-distance edges that don't represent real route segments.

---

## 9. Regional Line Geographic Scope

**Finding:** Regional lines (e.g., Line 18) extend beyond Tübingen city center.

**Example - Line 18:**

- Serves: Tübingen → Hagelloch → Poltringen → Rottenburg
- Poltringen coordinates: 48.53°N, 8.95°E (outside Tübingen bounding box 9.03-9.09°E)

**Decision:** Do not apply geographic bounding box filter for line-specific visualizations.

**Rationale:** Filtering to Tübingen city center would exclude legitimate route segments for regional lines.

---

_Last updated: January 31, 2026_
