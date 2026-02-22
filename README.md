# main.py — Synthetic 1D Benchmark for USKrig, USSNE, and CONE

## Purpose

This script compares three surrogate-based optimization algorithms on a synthetic 1D test problem with heteroscedastic noise. It measures each algorithm's ability to identify the conditionally optimal decision across a continuous state space, using the **Estimated Total Decision Loss (ETDL)** metric.

## Algorithms

### USKrig (Universal Stochastic Kriging)
- Gaussian Process surrogate fitted independently for each decision alternative.
- Samples design points uniformly at random from the (x, y) space.
- Uses batch replications (default batch size `l=5`) at each sampled point to estimate local mean and variance.

### USSNE (Universal Stochastic SNE)
- Shrinking Neighborhood Estimation (SNE) surrogate.
- Samples uniformly, one replication per point.
- The SNE surrogate computes a local average over observations within a shrinking neighborhood ball of radius controlled by the `Xi` parameter.

### CONE (Contextual Optimization with Non-parametric Estimation)
- Adaptive variant of USSNE that allocates simulation budget non-uniformly.
- Uses an **optimization-based sampling weight** `a(x,y)` to bias sampling toward informative regions of the (x, y) space.
- The sampling weight is computed by solving a 1D optimization problem (via `scipy.optimize.minimize_scalar`):
  - For the estimated-best decision `xhat` at state `y`, find `u_0` minimizing `h(u_0) = u_0^{-(1+xi)} + sum_{x != xhat} u_x^{-(1+xi)}`, where `u_x = (delta_x^2 - var_hat * u_0) / var_x`.
  - The resulting `a(x,y) = u*(x,y)^{-(1+xi)}` is clipped to `[M_L, M_U]`.
- Includes an optional **warm-up phase** that runs `warmup_K` uniform samples to calibrate `M_L` (10th percentile) and `M_U` (90th percentile) from observed sampling weights.
- During the main loop, candidate points are accepted via **rejection sampling** with acceptance probability `a(x,y) / M_U`.
- After training, `final_xstar(y)` returns the estimated optimal decision for any query state `y`.

## Test Problem (F1d)

A 1D problem with 3 decision alternatives (`X = {1, 2, 3}`) and a scalar state `y in [0, 2]`:

| Decision | Mean function `f(x, y)` | Noise std `sigma(x, y)` |
|----------|-------------------------|--------------------------|
| x = 1 | `10/(y+1)^2 * sin(e^(y+1))` | `0.5*(sin(16y) + 1.2)` |
| x = 2 | `0` (constant) | `0.5*(sin(8y) + 1.2)` |
| x = 3 | `-10/(y+0.8)^2 * sin(e^(y+0.8))` | `0.5*(sin(4y) + 1.2)` |

The conditionally optimal decision changes across `y`, making this a non-trivial contextual optimization problem. The noise is heteroscedastic (varies with both `x` and `y`).

## Configuration

Global parameters at the top of the file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metatrail` | 10 | Number of independent replications per algorithm |
| `Totalbudget` | 1000 | Total simulation budget per run |
| `TDLMCsize` | 20 | Grid resolution for ETDL estimation |

CONE-specific parameters (passed at instantiation):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Xi` | 1 | SNE shrinking speed |
| `upper` | 30 | Initial M_U |
| `lower` | 1 | Initial M_L |
| `warmup` | True | Enable warm-up calibration |
| `warmup_K` | 50 | Warm-up sample count |

## Running

```bash
python main.py
```

**Dependencies:** numpy, pandas, scipy, scikit-learn, matplotlib

**Runtime:** Scales linearly with `Totalbudget * metatrail`. With defaults (1000 budget, 10 metatrials), expect several minutes.

## Output

1. **TDLvObs.png** — Line plot of 1 - Total Decision Loss vs. cumulative observations for all three algorithms (black = USKrig, blue = USSNE, red = CONE). Higher is better.
2. **Console output** — Per-algorithm budget countdown and warm-up calibration values (`M_L`, `M_U`).
3. **Histograms** (when `metatrail == 1`) — `hist_y_1.png`, `hist_y_2.png`, `hist_y_3.png` showing the distribution of sampled `y` values for each decision alternative under CONE, revealing the adaptive allocation pattern.

## Importing CONE from Other Scripts

Since `Rand.seed(0)` is guarded under `if __name__ == "__main__"`, CONE can be imported without side effects:

```python
from main import CONE
```

The `onestage_optimizer.py` script uses this import to apply CONE to the call center staffing problem, passing `Y_weight` for multi-dimensional state normalization.

## Key Classes and Functions

| Name | Type | Description |
|------|------|-------------|
| `CONE` | Class | Main algorithm. Constructor accepts `TestProblem`, `totalbudget`, `Xi`, `upper`, `lower`, `Y_weight`, `warmup`, `warmup_K`. |
| `USSNE` | Class | Non-adaptive SNE baseline. Constructor: `TestProblem`, `totalbudget`, `Xi`. |
| `USKrig` | Class | GP-based baseline. Constructor: `TestProblem`, `l` (batch size), `totalbudget`. |
| `TestProblem` | Class | Wraps a test function. Fields: `X` (decision set), `Y` (state bounds as Nx2 array), `f` (mean function), `sigma` (noise function). |
| `ETDL()` | Function | Estimates Total Decision Loss over a grid of `y` values. |
| `YgridGenerator()` | Function | Creates an evenly spaced grid over the state space for ETDL evaluation. |
