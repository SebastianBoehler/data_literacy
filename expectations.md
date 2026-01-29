# Data Literacy Final Project — Specification & Checklist

## Goal

Produce a statistically sound, uncertainty-aware analysis and a concise 4-page paper that meets the Data Literacy course expectations.

The project emphasizes:

- estimation over testing
- distributions over single numbers
- uncertainty-aware visualization
- careful, honest interpretation

---

## 1. Core Principles (Must Follow)

- Treat analysis as estimation of distributional properties, not hypothesis confirmation.
- Prefer visualization + uncertainty over formal hypothesis testing.
- Use simple, interpretable quantities (probabilities, medians, quantiles).
- Claims must scale with evidence.
- Be explicit about limitations and non-causality.

---

## 2. Code Structure Requirements

### 2.1 Separation of Concerns

Structure code into clearly separated stages:

1. Data loading
2. Data cleaning / preprocessing
3. Statistical estimation
4. Visualization

Avoid:

- Monolithic notebooks
- Mixing preprocessing and plotting
- Copy-pasted code blocks

Prefer:

- Small, single-purpose functions
- Reusable plotting helpers
- Clear function naming

---

### 2.2 Reproducibility

- All results must be deterministic
- Fix random seeds (e.g. bootstrap)
- Code must run top-to-bottom without manual intervention
- No reliance on hidden notebook state

---

### 2.3 Explicit Design Choices

All choices must be explicit and documented:

- Bin widths
- Thresholds
- Category definitions
- Top-k selections

Use named constants. Avoid magic numbers.

---

## 3. Statistical Reasoning Rules

### 3.1 Estimation First

Primary outputs:

- Probabilities (e.g. P(delay > 0))
- Medians and quantiles
- Distributional comparisons

Avoid:

- Overreliance on means for skewed data
- Black-box models

---

### 3.2 Uncertainty Is Mandatory

Every estimated quantity must include uncertainty unless clearly unnecessary.

Approved methods:

- Bootstrap confidence intervals
- DKW confidence bands for ECDFs
- Sample size annotations

If uncertainty is missing, the result is incomplete.

---

### 3.3 Hypothesis Tests (Optional)

- Tests are not the default
- Use only for clear decision questions
- Always report effect sizes
- Never rely on p-values alone

---

## 4. Plotting Standards

### 4.1 Purpose

Every plot must answer a specific question:

> What quantity is being estimated or compared?

Remove plots without a clear purpose.

---

### 4.2 Preferred Plot Types

- ECDFs over histograms
- Quantiles over means
- Late rate over average delay
- Small multiples over overloaded plots

---

### 4.3 Design Rules

- Axes labeled with units
- Titles describe content, not code
- Color encodes meaning, not decoration
- Consistent color semantics across figures
- No chart junk (3D, unnecessary effects)

---

### 4.4 Uncertainty Visualization

- CI ribbons or error bars where appropriate
- DKW bands for ECDFs
- Explicit sample size for imbalanced groups

---

### 4.5 Figure Discipline

- Prefer few, strong figures
- Remove redundant plots
- Supplementary plots go to GitHub, not the paper

---

## 5. Paper Writing Guidelines

### 5.1 Tone & Language

Use:

- “we estimate”
- “is associated with”
- “suggests”
- “within uncertainty”

Avoid:

- “prove”
- “confirm”
- “clearly shows”
- Unqualified causal language

---

### 5.2 Abstract

- State the question
- Define the quantities estimated
- Summarize main qualitative findings
- Emphasize uncertainty-aware analysis

---

### 5.3 Introduction

- Motivate the real-world problem
- Explain why it is vague
- Describe how it is made statistically precise

---

### 5.4 Data & Methods

- Describe data and preprocessing
- Justify estimator choices
- Emphasize nonparametric, robust methods
- Explicitly mention observational nature of data

---

### 5.5 Results

For each result:

1. Define the quantity
2. Reference the figure
3. Interpret magnitude and uncertainty
4. Avoid causal claims

---

### 5.6 Discussion & Limitations

Mandatory section.

Include:

- Data limitations
- Missing variables
- Observational bias
- What findings are robust vs tentative

---

## 6. Figure Selection Strategy (4-Page Limit)

### Core Figures (Target: 4–5)

- Overall delay distribution (ECDF + uncertainty)
- Time-of-day effect with CI
- Spatial heterogeneity (stops or network)
- One representative weather effect

### Supplementary (GitHub)

- Additional weather variables
- Correlation heatmaps
- Effect size comparisons
- Extra rankings

---

## 7. Final Pre-Submission Checklist

### Code

- [ ] Modular, readable, deterministic
- [ ] Fixed random seeds
- [ ] No magic numbers

### Plots

- [ ] Each plot answers one question
- [ ] Uncertainty shown
- [ ] Units, labels, titles correct
- [ ] No redundancy

### Paper

- [ ] Precise definitions of quantities
- [ ] Effect sizes emphasized
- [ ] No overclaiming
- [ ] Limitations discussed
- [ ] Fits cleanly in 4 pages

---

## Guiding Principle

The goal is not to prove a hypothesis,  
but to learn something reliable about the data  
and communicate it clearly, cautiously, and honestly.
