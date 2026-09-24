# Agentic EMS — GPU ambulance relocation, driven by natural language

When an ambulance leaves its post for a call, it tears a hole in the city's
coverage map. Dispatchers patch that hole by hand, under time pressure, using
intuition. This project closes the loop automatically: a GPU mixed-integer
program continuously repositions idle units to maximize demand-weighted
coverage, and a local LLM lets a dispatcher steer it in plain English —
*"Major collision in M5V, roll three units now."*

Built in ~2 days on an **NVIDIA GB10 (DGX Spark)** at an NVIDIA hackathon, by a
team of three. Everything runs locally on the one box: no cloud inference, no
data leaving the machine.

## The whole stack is local and GPU-resident

```
 dispatch audio ──▶ Parakeet ASR ──▶ normalize ──┐
                                                  ├─▶ Nemotron 33B ──▶ Constraints
 typed command ───────────────────────────────────┘   (intent + parse)      │
                                                                            ▼
 Toronto open data ──▶ cuDF clean ──▶ cuGraph road graph ──▶ World      cuOpt MILP
 (centreline, stations,               (station↔FSA                     (relocation)
  paramedic incidents)                 travel-time matrix)                  │
                                                                            ▼
 discrete-event sim ◀── apply moves ◀── OptimizeResult ──▶ Nemotron explains why
```

| Layer | NVIDIA component | What it does |
|---|---|---|
| Data + graph | **cuDF, cuGraph** | Cleans city open data; builds the road network and the cached station↔FSA travel-time matrix |
| Optimization | **cuOpt 26.4** | Solves the relocation MILP on GPU via `cuopt.linear_programming` |
| Uncertainty | **CuPy** | Monte Carlo demand sampling for *expected* coverage across scenarios |
| Language | **Nemotron 3 33B** (local NIM) | Intent classification, command → solver constraints, and causal explanation of the result |
| Voice | **Parakeet ASR** (local service) | Spoken dispatch radio → text, on-GPU, audio never leaves the box |

## The optimization model

The core decision: given units that are currently free, which posting station
should each one move to?

**Variables** — `x[u,s] ∈ {0,1}` unit `u` posted at station `s`; `y[z] ∈ {0,1}`
zone `z` is covered.

**Maximize** `Σ_z demand[z]·y[z] − move_penalty · Σ_{u, s≠home[u]} x[u,s]`

The penalty term is what keeps the solution operationally sane — without it the
solver happily reshuffles the entire fleet every tick to buy a rounding error of
coverage.

**Subject to**

| | Constraint |
|---|---|
| A | `Σ_s x[u,s] = 1` — every available unit is posted exactly once |
| B | `y[z] − Σ_{u,s : cov[s,z]} x[u,s] ≤ 0` — a zone counts as covered only if some unit actually reaches it |
| C | `Σ_u x[u,s] ≤ capacity[s]` — station capacity |
| D | `Σ_{u, s≠home[u]} x[u,s] ≤ max_moves` — optional cap on churn |

Dispatcher instructions enter as bounds rather than new constraints:
`protect_zones` pins `y[z] = 1`, and `lock_units` / `force_station` pin the
corresponding `x[u,s]`. That means a natural-language command never changes the
shape of the program — only its box constraints — so the LLM cannot produce an
infeasible or malformed model.

Demand zones are **FSAs** (the first three postal-code characters). That was
less a modelling choice than a privacy constraint turned into a feature: the
public paramedic incident data strips street-level location and publishes only
the FSA, which hands you a zone partition for free.

## Language as a control surface

`agent.parse_command(text, world) → Constraints` classifies the operator's
intent before it does anything else:

- **emergency** — an active incident. Fills a dispatch: zone, unit count,
  reason, priority.
- **coverage** — a readiness request. Fills constraints and lets the optimizer
  rebalance.

Those are genuinely different operations, and conflating them was the bug that
took longest to find: a dispatcher shouting about a collision wants units *sent*,
not coverage *rebalanced*, and early versions treated both as the latter.

After the solve, `explain_result(result, world) → str` narrates causally — not
"3 units moved" but which unit covered which commanded zone and what it gave up
to do it. An optimizer that cannot explain itself does not get trusted by the
person who has to press the button.

## Design notes worth stealing

**Interfaces frozen in hour one.** [`CONTRACTS.md`](CONTRACTS.md) fixed every
data shape between the three lanes before anyone wrote code, so integration at
the end was plugging cables together instead of a merge crisis. The single
agreed function:

```python
re_optimize(state: State, world: World, constraints: Constraints | None) -> OptimizeResult
```

**Two CUDA environments, one repo.** cuGraph needed CUDA 13, cuOpt needed
CUDA 12. Rather than fight it, the lanes run in separate venvs and exchange a
`world.npz` on disk — the split is invisible across the contract boundary. NeMo
gets a third env (`~/asr-env`) so its dependency tree never touches the working
cuOpt stack.

**Everything degrades.** `cuOpt` is imported lazily, so the loop runs
coverage-only on a machine without a GPU; the Nemotron client falls back to a
rule-based parser and a template explanation when the NIM is offline. The whole
demo runs on a laptop.

**Never `inf`.** Unreachable travel times use a large finite sentinel (`1e6`).
cuOpt chokes on `inf`/`NaN`, and it does so deep inside the solve where the
error message tells you nothing.

## Running it

```bash
source ~/cuopt-env/bin/activate

python -m optimizer.demo                # end-to-end smoke test on mocks
python -m sim_agent.run                 # full loop, real world (needs cuOpt)
python -m sim_agent.run --mock          # frozen mock world
python -m sim_agent.run --no-optimize   # coverage-only, no GPU required
python -m sim_agent.selftest            # fast no-GPU checks

python webapp/server.py                 # deck.gl map UI
python bench_llm.py                     # ollama vs llama-server on the real prompt
```

`integration_check.py` verifies the three lanes still agree on the contracts;
`verify_rapids.py` confirms RAPIDS is live on the GPU.

## Layout

| Path | Lane | Contents |
|---|---|---|
| `optimizer/` | cuOpt | `milp.py` (model build), `reoptimize.py` (solve + decode + reasoning), `evaluate.py` (fast non-GPU coverage), `montecarlo.py` (CuPy scenarios) |
| `sim_agent/` | Sim + agent | `engine.py` (discrete-event sim, Poisson arrivals), `agent.py` (Nemotron client, parse + explain), `loop.py` (one decision tick) |
| `frontend/`, `webapp/` | UI | deck.gl map, FSA GeoJSON export, timeline generation |
| `audio/` | Voice | Parakeet ASR service + client, dispatch clip tooling |
| `contracts.py`, `CONTRACTS.md` | — | The frozen interfaces |

## Honest limitations

- **Coverage is a proxy.** The objective maximizes demand-weighted zone
  coverage, not response time or survival. Those correlate, but they are not the
  same thing, and a real deployment would need to optimize the outcome directly.
- **Travel times are static.** The cuGraph matrix is computed once and cached;
  there is no live traffic, no time-of-day variation, no weather.
- **The sim is not validated.** Poisson arrivals weighted by historical FSA
  demand are a reasonable first model, but nothing here was checked against real
  dispatch logs, so the simulated improvements are internally consistent rather
  than externally verified.
- **Two days.** Written at a hackathon. The contracts and fallbacks are solid;
  test coverage is `selftest.py` and `integration_check.py`, and that is it.

## Data

Toronto Open Data: Centreline (road geometry), Ambulance Station Locations
(posting candidates), Paramedic Services Incident Data (demand, by FSA).

## Credits

Three-person hackathon team, lanes split as described in `CONTRACTS.md`.
Note that the git history is authored from the shared GB10 workstation and does
not reflect individual contribution.
