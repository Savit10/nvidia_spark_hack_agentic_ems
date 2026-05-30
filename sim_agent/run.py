"""
End-to-end demo of the sim + agent lane (Person 3).

Run from the repo root in the cuOpt env:
    source ~/cuopt-env/bin/activate
    python -m sim_agent.run                       # real world (datasets/world.npz)
    python -m sim_agent.run --mock                # frozen mock world
    python -m sim_agent.run --no-optimize         # coverage only (no cuOpt)
    python -m sim_agent.run --ticks 8 --units 14

Drives the discrete-event sim through several decision ticks. On one tick it
injects an English operator command so you can see the agent translate it to
Constraints and explain the result.
"""
from __future__ import annotations

import argparse
import os
import sys

import contracts as C
from sim_agent import Loop, Simulation
from sim_agent.agent import nim_available

DATASETS = os.path.join(os.path.dirname(__file__), os.pardir, "datasets")


def load_real_world() -> C.World:
    sys.path.insert(0, DATASETS)
    from load_world import load_world, validate_world

    world = load_world(
        os.path.join(DATASETS, "world.npz"),
        os.path.join(DATASETS, "world_meta.json"),
    )
    validate_world(world)
    return world


def _fmt_cov(cm: C.CoverageMap) -> str:
    return (
        f"{cm['covered_demand_pct']*100:5.1f}% of demand covered, "
        f"{len(cm['gaps'])} gap zones"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person 3 sim + agent demo")
    ap.add_argument("--mock", action="store_true", help="use frozen mock world")
    ap.add_argument("--no-optimize", action="store_true", help="skip cuOpt")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--tick-min", type=float, default=15.0)
    ap.add_argument("--units", type=int, default=12)
    ap.add_argument("--calls-per-hour", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    world = C.mock_world() if args.mock else load_real_world()

    print("=" * 72)
    print(
        f"SIM + AGENT LANE  ({'mock' if args.mock else 'real'} world: "
        f"{world['S']} stations, {world['Z']} FSAs, "
        f"{world['threshold_min']:.0f}-min threshold)"
    )
    print(f"NIM agent: {'ONLINE' if nim_available() else 'offline -> rule-based fallback'}")
    print("=" * 72)

    sim = Simulation(
        world,
        n_units=args.units,
        calls_per_hour=args.calls_per_hour,
        seed=args.seed,
    )
    loop = Loop(sim, world, tick_min=args.tick_min, optimize=not args.no_optimize)

    # Inject an operator command partway through the run.
    sample_zone = world["fsa_index"][len(world["fsa_index"]) // 2]
    commands = {
        2: f"Make sure {sample_zone} stays covered and don't make more than 2 moves."
    }

    def on_tick(k: int, tr):
        print(f"\n--- tick {k}  (t={tr.t:.0f} min, {tr.calls_this_tick} calls) ---")
        print("  before :", _fmt_cov(tr.coverage_before))
        if tr.command:
            print("  command:", tr.command)
            print("  parsed :", tr.constraints)
        if tr.result is not None:
            print("  after  :", _fmt_cov(tr.result["coverage_after"]))
            print(
                f"  solver : {tr.result['status']} in "
                f"{tr.result['solve_time_ms']:.1f} ms, {tr.result['n_moves']} moves"
            )
        print("  agent  :", tr.explanation)

    loop.run(args.ticks, commands=commands, on_tick=on_tick)

    print("\n" + "=" * 72)
    k = sim.kpis()
    print("RUN KPIs")
    print(
        f"  calls={k['calls']}  served={k['served']}  missed={k['missed']}  "
        f"on-time={k['on_time_pct']*100:.0f}%"
    )
    print(
        f"  response: mean={k['mean_response_min']:.1f} min  "
        f"p90={k['p90_response_min']:.1f} min"
    )
    print("OK - sim + agent lane runs end-to-end.")


if __name__ == "__main__":
    main()
