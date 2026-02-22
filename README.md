# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **CONE (Contextual Optimizer through Neighborhood Estimation for Prescriptive Analytics)**.
Originated at NUS.

## Running the Code

No build system or test suite. Run individual scripts directly with Python 3.11:

```bash
python main.py                    # Synthetic 1D test function experiments
```

## Dependencies (no requirements.txt)

simpy, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

## Architecture

### Core Algorithms (classes appearing in multiple files with variations)

- **CONE** — The main CONE algorithm. Uses Shrinking Neighborhood Estimation (SNE) as a non-parametric surrogate for adaptive simulation budget allocation. Learns an optimal decision map x*(y) from state space to decision space.
- **USKrig** — Uniform Sampling Kriging baseline (Gaussian/Matérn surrogate).
- **USSNE** — Uniform Sampling SNE baseline.
- **TestProblem** — Wraps the simulation into a standard interface with decision space X, state space Y bounds, and a simulation function.

## Important Parameters (in `main.py`)
- `metatrail` number of meta trails to generate plot.
- `Totalbudget` number of total simlation trails.
- `TDLMCsize` sample size to estimate TDL.
