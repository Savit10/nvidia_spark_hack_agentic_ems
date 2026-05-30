# `optimizer/` — Person 2 (cuOpt + evaluation)

Coverage evaluation and the GPU relocation MILP. Public surface (frozen in
`contracts.py` / CONTRACTS.md):

```python
from optimizer import re_optimize, evaluate, expected_coverage
```

| function | signature | role |
|----------|-----------|------|
| `evaluate` | `(state, world, threshold_min=None) -> CoverageMap` | fast, non-cuOpt coverage snapshot for the map + KPI |
| `re_optimize` | `(state, world, constraints=None) -> OptimizeResult` | build + solve the relocation MILP on the GPU, return moves |
| `expected_coverage` | `(state, world, n_scenarios=...) -> dict` | GPU Monte Carlo expected coverage (the Spark story) |

## Environment

Runs in **`~/cuopt-env`** (CUDA-12: `cuopt-cu12` + cupy + scipy), NOT the repo
`.venv` (which is CUDA-13 cuGraph for Person 1). The two lanes exchange the
`world.npz` file, so the env split is invisible across the boundary.

```bash
source ~/cuopt-env/bin/activate
python -m optimizer.demo            # end-to-end smoke test on mocks
```

## Files

- `evaluate.py` — coverage mask, nearest-unit, CoverageMap. Pure numpy.
- `milp.py` — builds the cuOpt `DataModel` (vars/constraints/bounds). See the
  model docstring; mirrors CONTRACTS.md §7.
- `reoptimize.py` — `re_optimize` orchestration + solution decode.
- `montecarlo.py` — cuPy Monte Carlo expected coverage.
- `demo.py` — runnable smoke test against `contracts.mock_*`.

## Status / TODO

- [x] MILP solves on GB10; lock / force / protect / max_moves / capacity all enforced.
- [x] Graceful `Infeasible` and no-available-units handling.
- [x] GPU Monte Carlo (200k scenarios) runs.
- [x] **Integrated with real Person 1 data** (46 stations, 96 FSAs): coverage
      88.1% → 97.5% (+9.3 pts) with 4 moves. See `python integration_check.py`.
- [x] `move_penalty` is now **scale-invariant** (multiples of mean-zone demand,
      default 0.5) so it behaves the same regardless of FSA count.
- [ ] **BLOCKER on Person 1:** need a station→station travel matrix (S×S) in
      `world`. Without it (a) `move.eta_min` is haversine-estimated, and worse
      (b) the objective can't penalize *long* relocations — real runs pick
      24-38min moves that are operationally useless. This is the top fix.
- [ ] Once S×S exists: add a distance-weighted move cost (penalty ∝ relocation
      minutes) so the optimizer prefers nearby relocations.
- [ ] Wire to Person 3's real sim states (currently `integration_check.make_state`).
- [ ] Solver takes ~1.5s for 10 units × 46 stations — set a MIP gap/time limit
      in `SolverSettings` if the demo loop needs it faster.
