# Agent Prompt — Distributional & Tail-Focused Delay Analysis (Inspired by Chen et al., 2025)

## Context

This project analyzes public transport delay data for a Data Literacy course.
The goal is **not predictive modeling**, but **distributional, uncertainty-aware analysis**.

We take inspiration from:
Chen et al. (2025), Transportation Research Part C  
DOI: 10.1016/j.trc.2025.105000

The referenced work models bus delays probabilistically and emphasizes:

- distributional structure (not just means)
- tail behavior (extreme delays)
- condition-dependent effects (weather, time)
- probabilistic interpretation

We replicate these ideas **conceptually**, not via full Bayesian MCMC.

---

## High-Level Objective

Implement **distribution-first, tail-aware analyses** of delay data that:

- use simple, nonparametric estimators
- emphasize interpretability
- include uncertainty (bootstrap / ECDF bands)
- produce publication-quality plots

---

## Required Analysis Components

### 1. Delay Category Decomposition (Core Task)

Define **four delay categories**, inspired by the cited paper’s on-time states:

Use empirical quantiles (computed globally):

- extreme earliness: delay ≤ q05
- moderate earliness: q05 < delay ≤ 0
- moderate lateness: 0 < delay ≤ q95
- extreme lateness: delay > q95

Notes:

- Quantiles must be computed from the full dataset
- Thresholds must be explicit and documented
- Categories must be mutually exclusive and exhaustive

---

### 2. Category Composition by Condition

For each condition below, estimate the **proportion of trips in each delay category**:

Conditions to implement:

- precipitation bins (e.g. none / light / moderate / heavy)
- time-of-day bins (e.g. morning peak / off-peak)
- optional: weekday vs weekend

Estimation:

- proportions per category
- bootstrap confidence intervals for extreme lateness proportion

Plot:

- stacked bar charts (category composition)
- consistent color mapping across figures
- annotate sample sizes per group

---

### 3. Tail-Focused Metrics (Extreme Delays)

Implement tail-sensitive estimators:

- late rate above threshold: P(delay > q95)
- 90th and/or 95th percentile delay
- conditional tail metrics by weather or time

Uncertainty:

- bootstrap confidence intervals for tail quantities

Plot:

- bar or line plots with CI
- avoid hypothesis testing unless strictly necessary

---

### 4. Distributional Comparisons (No Parametric Assumptions)

Implement:

- ECDFs of delay for key condition comparisons
- optional: DKW confidence bands

Comparisons to include (choose minimal set):

- dry vs heavy precipitation
- peak vs off-peak hours

Avoid:

- fitting parametric distributions unless clearly justified
- overplotting too many conditions at once

---

## Explicit Non-Goals

Do NOT:

- implement Bayesian MCMC or Gaussian mixture inference
- optimize prediction accuracy
- rely on p-values as primary results
- make causal claims

Focus on:

- estimation
- uncertainty
- interpretability

---

## Code Structure Requirements

- Separate functions for:
  - category assignment
  - quantile computation
  - bootstrap CI estimation
  - plotting
- Fix random seeds for bootstrap
- No magic numbers (all thresholds named)
- Deterministic, rerunnable outputs

---

## Plotting Requirements

- Academic, publication-style plots (Matplotlib)
- Axes labeled with units
- Titles describe quantities, not code
- Color encodes meaning consistently
- Include uncertainty where relevant
- Avoid chart junk

---

## Expected Outputs

- Clean functions to compute delay categories
- At least one stacked category plot
- At least one tail-focused plot (quantile or extreme-late rate)
- ECDF comparison plot for a key condition
- Figures suitable for inclusion in a 4-page paper

---

## Interpretation Guidelines

When generating text or comments:

- Use language like “we estimate”, “is associated with”, “suggests”
- Explicitly mention uncertainty
- Emphasize effect sizes over significance
- Acknowledge observational limitations

---

## Guiding Principle

The objective is to **characterize delay distributions and their tails**,  
not to prove hypotheses or fit complex models.

All outputs should support cautious, transparent interpretation.
