# Codex vs Claude Adjudication

## Scope
Adjudication of claims made in `claude_response_followup.md`, cross-checked against current code and historical run outputs in `mujoco_refactor/results`.

## Verdict Table

### 1) "Velocity dominates ~92% of cost"
- Verdict: `ACCEPT`
- Evidence: Recomputed on run `mujoco_refactor/results/20260216T105746_sig_0-5_CMAES_warm_BEST-SO-FAR/multi_optimization_results.csv`.
- Result: Non-failure rows show mean contribution fractions approximately `velocity 0.920`, `variance 0.056`, `lateral 0.019`, `yaw 0.004`, `tumble ~0`, `pitch ~0`.

### 2) "scene2_f10 dominates best-point cost"
- Verdict: `ACCEPT`
- Evidence: Best row (`id=7c8c45ce`) decomposition from `cost_<ref>` columns.
- Result: `scene2_f10` contributes `0.3431` out of `0.5772` per-reference subtotal (~59%).

### 3) "Dead-zone refs still influence cost through variance"
- Verdict: `ACCEPT`
- Evidence: `mujoco_refactor/optimizer.py:408-418` computes variance from raw relative errors `(sim-target)/target`, independent of dead-zone truncation used in per-ref velocity term.

### 4) "Seed collision issue has been fixed"
- Verdict: `ACCEPT`
- Evidence: `mujoco_refactor/optimizer.py:84-85` and `mujoco_refactor/optimizer.py:254-260` now use collision-free index-based seed mapping.

### 5) "CMA remainder is documented but not fixed"
- Verdict: `ACCEPT`
- Evidence: `mujoco_refactor/config.py:97-100` documents limitation; `mujoco_refactor/optimizer.py:677-680` still ignores `n_points` in CMA `ask` wrapper.

### 6) "dof_damping range in earlier Claude critique was outdated"
- Verdict: `ACCEPT`
- Evidence: Active range is `Real(7e-12, 7e-9, "log-uniform", name="dof_damping")` in `mujoco_refactor/config.py:224`.

### 7) "sliding friction bests are not only 0.005–0.009"
- Verdict: `ACCEPT`
- Evidence: Historical best rows include values around `0.035` and `0.053` in `mujoco_refactor/results/*/optimization_bests.csv`.

### 8) "Velocity dominance is feature, not bug"
- Verdict: `PARTIAL`
- Rationale: This can be intentional for system ID, but remains a design risk unless explicitly declared objective policy.
- Technical nuance: Current objective pressure is highly concentrated in a few references (especially `scene2_f10`) in top solutions, which can reduce balanced fit across conditions.

### 9) "scene2_f10 mismatch likely structural"
- Verdict: `PARTIAL`
- Evidence: Across all 9600 evals, `scene2_f10` can be matched very closely in absolute error, but those points usually have poor global objective.
- Conclusion: Structural impossibility is not proven; currently best explained as multi-objective tradeoff under current weighting/dead-zone/variance design.

### 10) "P3 should be P1: fix magnetic_moment_fudge=1.0 and fit only field"
- Verdict: `PARTIAL / CAUTION`
- Evidence: Local A/B around best params (holding `kp_mag` product constant) showed substantial degradation when forcing `moment_fudge=1.0`.
- Result snapshot: baseline total ~`0.84` vs forced-moment total ~`3.09` in targeted test.
- Conclusion: Could be a principled physics-first choice, but not a free improvement; must be treated as explicit constraint tradeoff.

### 11) "solimp_dmin >= solimp_dmax invalid region wastes budget"
- Verdict: `PARTIAL`
- Confirmed facts:
  - Historical frequency is correct: `794/9600` (~`8.27%`) where `dmin >= dmax`.
  - Top 100 contains zero such points.
  - Median cost is worse for those points.
- Caution:
  - Calling this strictly "invalid for MuJoCo" is too strong without formal MuJoCo constraint proof.
  - Runtime tests show MuJoCo accepts and runs with `dmin > dmax` assignments.
- Better framing: empirically poor / low-value region under this objective, likely worth constraining by modeling policy.

## Additional Critical Corrections

1. Library vs wrapper limitation
- Important correction: pycma supports `es.ask(number=k)`.
- Current limitation is in local wrapper (`mujoco_refactor/optimizer.py:677-680`), not a hard pycma limitation.

2. Priority of actions
- Highest-confidence practical actions now:
  1. Decide and document objective policy (velocity-first vs balanced fit).
  2. Fix CMA wrapper/remainder behavior explicitly.
  3. Decide whether to constrain `dmin/dmax` ordering by policy.
  4. Run controlled identifiability experiment before hard-fixing magnetic moment.

## Final Adjudication
Claude follow-up is strong and materially useful. Most claims are correct. The main adjustments are:
- avoid over-stating `dmin >= dmax` as a hard MuJoCo-invalid condition,
- avoid assuming `moment_fudge=1.0` is a guaranteed improvement,
- separate library behavior from current wrapper behavior for CMA ask/remainder handling.
