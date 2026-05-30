"""
Person 1 -> Person 2 integration check.

Loads the REAL world.npz from Person 1 and runs the optimizer lane against it,
with a hand-built fleet state (the sim/Person 3 will supply real states later).

    cd ~/Desktop/nvidia-spark && python integration_check.py
"""
import sys
import time
from pathlib import Path

# Person 1's loader lives in datasets/
sys.path.insert(0, str(Path(__file__).parent / "datasets"))
from load_world import load_world, validate_world          # noqa: E402

from optimizer import evaluate, expected_coverage, re_optimize  # noqa: E402

DS = Path(__file__).parent / "datasets"


def make_state(world, n_units=12, busy=2):
    """Scatter n_units across stations (evenly spaced s-indices); mark a few busy."""
    S = world["S"]
    step = max(1, S // n_units)
    stations = list(range(0, S, step))[:n_units]
    units = []
    for i, s in enumerate(stations):
        units.append({
            "unit_id": f"AMB_{i:02d}",
            "status": "busy" if i < busy else "available",
            "station": s,
            "busy_until": 145.0 if i < busy else None,
        })
    return {"t": 120.0, "units": units}


def main():
    world = load_world(str(DS / "world.npz"), str(DS / "world_meta.json"))
    validate_world(world)
    print(f"loaded real world: S={world['S']} stations, Z={world['Z']} FSAs, "
          f"threshold={world['threshold_min']}min\n")

    state = make_state(world, n_units=12, busy=2)
    avail = sum(u["status"] == "available" for u in state["units"])
    print(f"fleet: {len(state['units'])} units ({avail} available, "
          f"{len(state['units'])-avail} busy)\n")

    before = evaluate(state, world)
    print(f"[before]  demand-covered = {before['covered_demand_pct']*100:.1f}%   "
          f"gaps = {len(before['gaps'])} FSAs")

    t0 = time.time()
    r = re_optimize(state, world, {"max_moves": 5, "move_penalty": 0.05})
    wall = (time.time() - t0) * 1000
    print(f"[re_optimize] status={r['status']}  moves={r['n_moves']}  "
          f"solver={r['solve_time_ms']:.1f}ms  wall={wall:.0f}ms")
    for m in r["moves"]:
        f, t = world["station_index"][m["from"]], world["station_index"][m["to"]]
        print(f"    {m['unit_id']}: {f} -> {t}  (~{m['eta_min']}min)  [{m['reason']}]")
    print(f"[after]   demand-covered = {r['coverage_after']['covered_demand_pct']*100:.1f}%   "
          f"gaps = {len(r['coverage_after']['gaps'])} FSAs")
    gained = (r["coverage_after"]["covered_demand_pct"]
              - r["coverage_before"]["covered_demand_pct"]) * 100
    print(f"          delta = {gained:+.1f} pts of demand coverage\n")

    mc = expected_coverage(state, world, n_scenarios=200_000)
    print(f"[montecarlo] expected={mc['expected_coverage_pct']*100:.1f}%  "
          f"p5={mc['p5']*100:.1f}%  p95={mc['p95']*100:.1f}%  "
          f"({mc['n_scenarios']:,} scenarios, gpu={mc['on_gpu']})")
    print("\nOK - optimizer lane runs on REAL Person 1 data.")


if __name__ == "__main__":
    main()
