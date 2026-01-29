# Data Source Change: Actual Observed Delays vs Estimated Delays

**Date**: 2026-01-29

## Summary

The notebook now uses **actual observed delays** from trip data instead of **estimated delays** from departure data.

## Why This Matters

| Data Source | Delay Type | Issue |
|-------------|------------|-------|
| Departure data (`all_departure_data` - old) | **Estimated** | `estimated_time` is a forecast for future departures - looking into the future |
| Trip data (`phase == "previous"`) | **Actual observed** | Bus has already passed these stops - real measurements |

## Technical Change

**Before**: Loaded departure data directly, which contains `delay_minutes = estimated_time - planned_time`. The `estimated_time` is a prediction for when the bus will depart in the future.

**After**: 
1. Load trip data and filter for `phase == "previous"` (stops the bus has already visited)
2. Deduplicate by keeping latest observation per `(journey_ref, operating_day_ref, stop_point_ref)`
3. Use `departure_delay_minutes` as `delay_minutes` - these are actual observed delays
4. Join weather data from departure data by timestamp

## Paper Sections to Update

- [ ] **Data Collection / Methodology**: Explain that we use trip-level data filtered for previous stops to get actual observed delays
- [ ] **Data Description**: Update any references to "departure data" to clarify we're using trip data for delay analysis
- [ ] **Limitations**: May want to note that this approach gives us fewer data points but more accurate delay measurements

## Key Numbers Change

The delay statistics will likely differ slightly because:
- Estimated delays (old) tend to be more optimistic/smoothed
- Actual observed delays (new) capture real-world variability

Run the notebook to get updated statistics for the paper.
