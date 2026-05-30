"""
End-to-end smoke test of the optimizer lane against the FROZEN mocks.

Run (from repo root, in the cuOpt env):
    source ~/cuopt-env/bin/activate
    python -m optimizer.demo
"""
from __future__ import annotations

from copy import deepcopy

import contracts as C
from optimizer import (
    evaluate,
    expected_coverage,
    rank_plans,
    re_optimize,
    re_optimize_robust,
)


def _fmt_cov(world, cm):
    return f"{cm['covered_demand_pct'] * 100:5.1f}%  covered={cm['covered']}  gaps={cm['gaps']}"


def main():
    world = C.mock_world()
    state = C.mock_state()
    constraints = C.mock_constraints()

    print("=" * 70)
    print("OPTIMIZER LANE SMOKE TEST  (mock world: "
          f"{world['S']} stations, {world['Z']} FSAs)")
    print("=" * 70)

    before = evaluate(state, world)
    print("\n[evaluate] current coverage:")
    print("  ", _fmt_cov(world, before))

    print("\n[re_optimize] constraints:", constraints)
    result = re_optimize(state, world, constraints)
    print(f"  status        : {result['status']}")
    print(f"  solve time    : {result['solve_time_ms']:.2f} ms")
    print(f"  objective     : {result['objective']:.4f}")
    print(f"  n_moves       : {result['n_moves']}")
    for m in result["moves"]:
        f = world["station_index"][m["from"]]
        t = world["station_index"][m["to"]]
        print(f"    move {m['unit_id']}: {f} -> {t}  (~{m['eta_min']} min)  [{m['reason']}]")
    print("  coverage before:", _fmt_cov(world, result["coverage_before"]))
    print("  coverage after :", _fmt_cov(world, result["coverage_after"]))

    print("\n[expected_coverage] GPU Monte Carlo on current placement:")
    mc = expected_coverage(state, world, n_scenarios=200_000)
    print(f"  mean={mc['expected_coverage_pct']*100:.1f}%  "
          f"p5={mc['p5']*100:.1f}%  p95={mc['p95']*100:.1f}%  "
          f"scenarios={mc['n_scenarios']:,}  on_gpu={mc['on_gpu']}")

    print("\n[rank_plans] GPU ranking: current vs an unconstrained re-optimize")
    free = re_optimize(state, world, None)
    after = deepcopy(state)
    by_id = {u["unit_id"]: u for u in after["units"]}
    for m in free["moves"]:
        by_id[m["unit_id"]]["station"] = m["to"]
    rk = rank_plans([state, after], world, n_scenarios=200_000,
                    labels=["current", "optimized"])
    for d in rk["ranking"]:
        print(f"  {d['label']:10s} exp={d['expected_coverage_pct']*100:5.1f}%  "
              f"p5={d['p5']*100:5.1f}%  p95={d['p95']*100:5.1f}%")

    print("\n[re_optimize_robust] cuOpt BatchSolve over a demand-scenario portfolio")
    rr = re_optimize_robust(state, world, None, n_scenarios=8, n_score=200_000)
    rb = rr["robust"]
    print(f"  chosen plan   : {rb['chosen_plan']}  (exp={rb['expected_coverage_pct']*100:.1f}%, "
          f"p5={rb['p5']*100:.1f}%)")
    print(f"  portfolio     : solved {rb['n_scenarios_solved']}+1 MILPs in one BatchSolve "
          f"[{rb['batch_solve_ms']:.0f} ms], scored {rb['n_plans_scored']} plans")

    print("\nOK - optimizer lane runs end-to-end on mocks.")


if __name__ == "__main__":
    main()
