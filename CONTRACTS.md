# Integration Contracts

The whole point of this doc: **agree the data shapes in hour one so all three
people build against fixed interfaces and integration at the end is just
plugging cables together.** Each component talks to the next only through the
objects defined here. If you need to change a shape, change it *here first* and
tell the other two.

```
                 world (static, built once)
                        │
   ┌────────────┐   ┌───────────┐   ┌──────────────────────┐
   │  Person 1  │──▶│ Person 2  │──▶│      Person 3         │
   │ Data+Graph │   │  cuOpt    │   │  Sim + Nemotron LLM   │
   │  (RAPIDS)  │◀──│ optimize  │◀──│  (orchestrator)       │
   └────────────┘   └───────────┘   └──────────────────────┘
        world          re_optimize(state, world, constraints)
                              │
                       returns OptimizeResult
```

The single function everyone agrees on:

```python
re_optimize(state: State, world: World, constraints: Constraints | None) -> OptimizeResult
```

- **Person 1** produces `World` (static, loaded once).
- **Person 3** owns the loop: builds `State` from the sim, asks the LLM for
  `Constraints`, calls `re_optimize`, feeds the result to the LLM for an
  explanation, applies the moves back into the sim.
- **Person 2** owns `re_optimize`, `evaluate`, and the Monte Carlo scorer.

Everything below is plain JSON-serializable Python (dicts, lists, numpy arrays).
No custom classes required at the boundary — use TypedDicts/dataclasses if you
like, but the **field names and shapes are the contract**.

---

## 0. Indexing convention (read this first)

Everything is keyed by **integer index**, with parallel lookup lists to recover
the human IDs. This is what lets cuOpt and cuGraph work on dense arrays.

- Zones (FSAs):   index `z` in `0 .. Z-1`
- Stations:       index `s` in `0 .. S-1`
- Units:          index `u` in `0 .. U-1` (units are dynamic; built per State)

`world["fsa_index"][z]` → `"M5V"`, `world["station_index"][s]` → `"STN_017"`.
Never pass a bare FSA string into the matrix — convert to `z` first.

---

## 1. `World` — Person 1 → everyone (static, built once)

cuDF cleaning + cuGraph travel-time matrix, computed once and **cached to disk**.
Persisted as `world.npz` (arrays) + `world_meta.json` (the index lists & scalars).

```python
World = {
    # --- index lookups (position = integer index) ---
    "fsa_index":     list[str],   # len Z, e.g. ["M4B", "M5V", ...]
    "station_index": list[str],   # len S, e.g. ["STN_001", ...]

    # --- metadata, keyed by index as string in JSON ---
    "fsa_meta": {                 # one entry per z
        z: {"fsa": str, "lat": float, "lon": float, "demand_weight": float}
    },
    "station_meta": {             # one entry per s
        s: {"station_id": str, "name": str, "lat": float, "lon": float, "capacity": int}
    },

    # --- the heavy arrays (in world.npz) ---
    "travel_time":    np.ndarray, # shape (S, Z), float32, minutes station→zone-centroid
    "coverage":       np.ndarray, # shape (S, Z), bool,  == (travel_time <= threshold_min)
    "station_travel": np.ndarray, # shape (S, S), float32, minutes station→station (drive
                                  #   between posting points). Diagonal == 0. Person 2 uses
                                  #   this for accurate move.eta_min AND a distance-weighted
                                  #   move cost so it prefers nearby relocations.

    # --- scalars ---
    "threshold_min": float,       # response-time threshold, e.g. 9.0
    "Z": int, "S": int,
}
```

**Contract guarantees Person 1 must hold:**
- `travel_time[s][z]` (and `station_travel[s][s']`) is always finite (use a big
  number, e.g. `1e6`, for unreachable — never `inf` or `NaN`; cuOpt chokes on
  non-finite). `station_travel` diagonal is exactly `0.0`.
- `demand_weight` is normalized so `sum(demand_weight) == 1.0` (it's a share of
  total expected calls). Person 2 multiplies coverage by this.
- `coverage` is exactly `travel_time <= threshold_min` — don't let them drift.
- Row order of `travel_time` matches `station_index`; col order matches `fsa_index`.

**Loader Person 2 & 3 call:**
```python
world = load_world("world.npz", "world_meta.json")  # Person 1 ships this fn
```

---

## 2. `State` — Person 3 → Person 2 (live snapshot each step)

The dynamic part: where the units are right now and which are free. The sim
produces this; the matrix/coverage live in `world`, not here.

```python
State = {
    "t":     float,               # sim clock (minutes since start), for logging
    "units": [
        {
            "unit_id":      str,  # "AMB_07"
            "status":       str,  # "available" | "busy"
            "station":      int,  # s-index where it's currently posted/based
            "busy_until":   float | None,  # sim time it frees up (None if available)
        },
        ...
    ],
}
```

- Only `status == "available"` units are candidates for relocation. Busy units
  are fixed where they are (still count for coverage if you want — agree this;
  default: busy units do **not** provide coverage).
- `station` is an `s`-index into `world["station_index"]`.

---

## 3. `Constraints` — Person 3 (LLM) → Person 2 (optional overrides)

This is the bridge from natural language to the optimizer. Nemotron parses an
operator's English command into **this exact object**. All fields optional; a
missing field means "use the default." `re_optimize(..., constraints=None)` runs
the plain model.

```python
Constraints = {
    "lock_units":    list[str],         # unit_ids that must NOT move
    "force_station": {str: int},        # unit_id -> s-index it must be posted at
    "max_moves":     int | None,        # cap on number of relocations
    "protect_zones": list[str],         # FSA codes that MUST end up covered (hard)
    "threshold_min": float | None,      # override response threshold for this run
    "move_penalty":  float | None,      # weight on each move in the objective
}
```

**Defaults (when field absent):** `lock_units=[]`, `force_station={}`,
`max_moves=None`, `protect_zones=[]`, `threshold_min=world["threshold_min"]`,
`move_penalty=0.1`.

This object is the **LLM's entire output surface for commands** — see §6.

---

## 4. `OptimizeResult` — Person 2 → Person 3 (the answer)

```python
OptimizeResult = {
    "moves": [
        {
            "unit_id":  str,
            "from":     int,     # s-index
            "to":       int,     # s-index
            "eta_min":  float,   # travel_time[from->to] for this unit's drive
            "reason":   str,     # short tag, e.g. "cover_gap:M6K"
        },
        ...
    ],
    "coverage_before": CoverageMap,   # see §5
    "coverage_after":  CoverageMap,
    "objective":       float,         # solver objective value
    "solve_time_ms":   float,         # from cuOpt get_solve_time()
    "status":          str,           # "Optimal" | "Feasible" | "Infeasible" | "Error"
    "n_moves":         int,
}
```

If `status == "Infeasible"` (e.g. impossible `protect_zones`), `moves` is `[]`
and Person 3 should have the LLM explain *why* using `coverage_before.gaps`.

---

## 5. `CoverageMap` — produced by Person 2's `evaluate()`

```python
CoverageMap = {
    "covered_demand_pct": float,      # demand-weighted % of zones covered (0..1) — the headline KPI
    "covered":            list[str],  # FSA codes covered by >=1 available unit within threshold
    "gaps":               list[str],  # demand zones with NO available unit in threshold
    "per_zone": {                     # full detail for the map render
        str: {                        # key = FSA code
            "covered":           bool,
            "nearest_unit_min":  float,    # travel time of closest available unit
            "demand_weight":     float,
        }
    },
}
```

`evaluate(state, world) -> CoverageMap` is a pure, fast (non-cuOpt) function —
Person 3 can call it directly to color the map without running the optimizer.

---

## 6. The LLM (Nemotron) contract — Person 3

Two directions. Both go through strict JSON so there's no fuzzy parsing.

### 6a. Command → `Constraints`  (natural language in, JSON out)

Operator types: *"Keep AMB_03 where it is and make sure Liberty Village stays
covered, but don't make more than 3 moves."*

LLM must emit exactly a `Constraints` object (§3). Use Nemotron function/tool
calling (or constrained JSON) with this schema:

```json
{
  "name": "set_constraints",
  "parameters": {
    "type": "object",
    "properties": {
      "lock_units":    {"type": "array", "items": {"type": "string"}},
      "force_station": {"type": "object", "additionalProperties": {"type": "integer"}},
      "max_moves":     {"type": ["integer", "null"]},
      "protect_zones": {"type": "array", "items": {"type": "string"}},
      "threshold_min": {"type": ["number", "null"]},
      "move_penalty":  {"type": ["number", "null"]}
    }
  }
}
```

The LLM needs the **name→id maps** to resolve "Liberty Village" → `"M6K"` and
"the downtown unit" → `"AMB_03"`. Person 1 ships a small glossary:
`world_meta.json` already has FSA codes + station names; build a
`{friendly_name: code}` dict from it and put it in the system prompt.

### 6b. `OptimizeResult` → explanation  (JSON in, English out)

LLM receives the `OptimizeResult` (§4) plus the name maps and produces a 2-4
sentence operator-facing explanation. Prompt template input:

```python
{
    "moves":   result["moves"],            # with ids resolved to names
    "before":  result["coverage_before"]["covered_demand_pct"],
    "after":   result["coverage_after"]["covered_demand_pct"],
    "gaps_healed": before.gaps - after.gaps,   # list of FSAs
    "n_moves": result["n_moves"],
}
```

Output: plain string. No JSON needed on the way out.

---

## 7. How Person 2 maps the model into cuOpt (reference)

So Person 2 isn't guessing the API. This is the **real installed cuOpt 26.4 LP/
MILP interface** (`cuopt.linear_programming`). Verified working on the GB10.

**Decision variables** (flattened into one vector for cuOpt):
- `x[u, s]` ∈ {0,1} — unit `u` posted at station `s`  (U·S binaries)
- `y[z]`    ∈ {0,1} — zone `z` is covered             (Z binaries)

Index map: `var_index(u, s) = u*S + s`; `y` vars come after, at `U*S + z`.

**Objective** (maximize):
```
maximize  Σ_z demand[z] * y[z]  −  move_penalty * Σ_{u,s != home(u)} x[u,s]
```

**Constraints:**
- Each available unit at exactly one station:  `Σ_s x[u,s] = 1`           (one row per u, type 'E')
- Coverage link: zone covered only if some unit sits at a covering station:
  `y[z] ≤ Σ_{u,s : coverage[s,z]} x[u,s]`                                  (type 'L')
- `protect_zones`: force `y[z] = 1`  (bound or 'E' row)
- `max_moves`:    `Σ moves ≤ max_moves`                                   (type 'L')
- `lock_units` / `force_station`: fix `x[u,s]` via variable bounds.

**The actual cuOpt calls** (confirmed API):
```python
from scipy.sparse import csr_matrix
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.solver_settings import SolverSettings
from cuopt.linear_programming import solver

dm = DataModel()
dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)  # A: (n_rows, n_vars) CSR
dm.set_row_types(row_types)                 # np.array of 'E'/'L'/'G', one per row
dm.set_constraint_bounds(rhs)               # right-hand side vector
dm.set_objective_coefficients(obj)          # length n_vars
dm.set_variable_types(np.array(['I']*n_vars))   # 'I' integer, 'C' continuous
dm.set_variable_lower_bounds(lb)            # zeros
dm.set_variable_upper_bounds(ub)            # ones (use to pin locked units)
dm.set_maximize(True)

sol = solver.Solve(dm, SolverSettings())
status = sol.get_termination_reason()       # "Optimal" / ...
x      = sol.get_primal_solution()          # np.array length n_vars
obj    = sol.get_primal_objective()
ms     = sol.get_solve_time() * 1000
```

Decode `moves` by reading back `x[u*S + s] > 0.5` and comparing to each unit's
`state` home station.

> **The Monte Carlo / Spark story (Person 2):** wrap `evaluate()` over many
> sampled demand realizations on the GPU (cuPy) to report *expected* coverage,
> not just current coverage. Same `World`, no new contract — it consumes
> `world["travel_time"]` + sampled demand and returns a distribution of
> `covered_demand_pct`.

---

## 8. Mock data — so everyone builds in parallel from minute one

Drop this in `mocks.py`. Person 2 builds against `mock_world()` + `mock_state()`
while Person 1 wrangles real data; Person 3 builds against `mock_result()` while
Person 2 writes the solver.

```python
import numpy as np

def mock_world(S=5, Z=8, seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    tt = rng.uniform(2, 20, size=(S, Z)).astype("float32")
    thr = 9.0
    dem = rng.uniform(0.05, 1.0, size=Z); dem /= dem.sum()
    return {
        "fsa_index":     [f"M{i}A" for i in range(Z)],
        "station_index": [f"STN_{i:03d}" for i in range(S)],
        "fsa_meta":      {z: {"fsa": f"M{z}A", "lat": 43.6+0.01*z, "lon": -79.4-0.01*z,
                              "demand_weight": float(dem[z])} for z in range(Z)},
        "station_meta":  {s: {"station_id": f"STN_{s:03d}", "name": f"Station {s}",
                              "lat": 43.65+0.01*s, "lon": -79.38-0.01*s, "capacity": 4}
                          for s in range(S)},
        "travel_time":   tt,
        "coverage":      tt <= thr,
        "threshold_min": thr, "Z": Z, "S": S,
    }

def mock_state():
    return {"t": 120.0, "units": [
        {"unit_id": "AMB_00", "status": "available", "station": 0, "busy_until": None},
        {"unit_id": "AMB_01", "status": "available", "station": 1, "busy_until": None},
        {"unit_id": "AMB_02", "status": "busy",      "station": 2, "busy_until": 145.0},
    ]}

def mock_constraints():
    return {"lock_units": ["AMB_00"], "force_station": {}, "max_moves": 2,
            "protect_zones": ["M3A"], "threshold_min": None, "move_penalty": 0.1}

def mock_result():
    return {
        "moves": [{"unit_id": "AMB_01", "from": 1, "to": 3, "eta_min": 6.2,
                   "reason": "cover_gap:M3A"}],
        "coverage_before": {"covered_demand_pct": 0.71, "covered": ["M0A","M1A"],
                            "gaps": ["M3A","M6A"], "per_zone": {}},
        "coverage_after":  {"covered_demand_pct": 0.89, "covered": ["M0A","M1A","M3A"],
                            "gaps": ["M6A"], "per_zone": {}},
        "objective": 0.89, "solve_time_ms": 14.3, "status": "Optimal", "n_moves": 1,
    }
```

---

## 9. Hour-one checklist

- [ ] All three import `mocks.py` and agree the field names above are frozen.
- [ ] Person 1 commits `load_world()` signature + a tiny real `world.npz` ASAP
      (even 5 stations / 8 FSAs from real data unblocks everyone).
- [ ] Person 2 commits `re_optimize(state, world, constraints) -> OptimizeResult`
      and `evaluate(state, world) -> CoverageMap` stubs returning mock shapes.
- [ ] Person 3 commits the loop calling those stubs + the two LLM JSON schemas.
- [ ] Map render (folium/pydeck) reads only `CoverageMap.per_zone` + `world` meta.
