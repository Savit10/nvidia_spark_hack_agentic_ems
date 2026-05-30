# sim_agent — Person 3 (simulation + agent + integration)

The center of the loop. Owns the discrete-event simulation, the Nemotron NIM
agent, and the wiring that drives `sim → evaluate → re_optimize → agent`.

Consumes the frozen contracts only: `World` (Person 1), `evaluate` +
`re_optimize` (Person 2). See `../CONTRACTS.md`.

## Files

| file | what |
|------|------|
| `engine.py` | discrete-event sim. Poisson calls (spatial = FSA `demand_weight`), nearest-available dispatch, busy/free state machine. Emits contract-shaped `State`; applies relocation `moves`. |
| `agent.py`  | Nemotron NIM client (stdlib `urllib`). `parse_command(text, world) → Constraints` (§6a) and `explain_result(result, world) → str` (§6b). Falls back to a rule-based parser + template when the NIM is offline. |
| `loop.py`   | `Loop` — one decision tick: advance sim, evaluate, parse command, re_optimize, explain, apply moves. cuOpt is imported lazily so it degrades to coverage-only without a GPU. |
| `run.py`    | end-to-end CLI demo. |
| `selftest.py` | fast no-GPU checks (sim + rule parser + explanation). |

## Run

```bash
source ~/cuopt-env/bin/activate            # numpy + cuOpt
cd <repo root>                             # so `import contracts` / `optimizer` resolve

python -m sim_agent.run                    # real world, full loop (needs cuOpt)
python -m sim_agent.run --mock             # frozen mock world
python -m sim_agent.run --no-optimize      # coverage-only, no GPU needed
python -m sim_agent.selftest               # no-GPU unit checks
```

## NIM (Nemotron) wiring

The agent speaks the OpenAI-compatible REST API. Point it at a local NIM via env:

```bash
export NIM_BASE_URL=http://localhost:8000/v1
export NIM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
```

If `NIM_BASE_URL` is unreachable, `parse_command` / `explain_result` silently use
the deterministic fallback, so the demo never blocks on the model being up.

## What still depends on others

- Move ETAs use a haversine station→station estimate — Person 1's note says
  `World` has no S×S matrix yet. Swap in their matrix when it lands (one spot:
  `optimizer/reoptimize.py:_haversine_min`).
- The agent glossary resolves **station names** and **FSA codes**. Friendly
  neighbourhood names ("Liberty Village" → M6K) need a `{name: fsa}` dict from
  Person 1 if we want them in the demo script.
