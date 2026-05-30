"""
re_optimize — Person 2's single public entry point (CONTRACTS.md).

    re_optimize(state, world, constraints) -> OptimizeResult

Person 3 calls this each time the sim wants a relocation decision. It:
  1. evaluates current coverage,
  2. builds + solves the relocation MILP on the GPU (cuOpt),
  3. decodes the unit->station assignment into a list of moves,
  4. evaluates the post-move coverage,
  5. returns the OptimizeResult shape the LLM explains.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import contracts as C
from optimizer.evaluate import evaluate
from optimizer.milp import build_milp


def _reloc_min(world: C.World, src: int, dst: int) -> float:
    """Real graph-based station->station relocation drive time (minutes), from
    Person 1's cuGraph station_travel matrix (diagonal 0)."""
    return float(world["station_travel"][src, dst])


def _decode(x, build, state: C.State, world: C.World,
            coverage_before: C.CoverageMap, rc: Optional[dict] = None):
    """Read a cuOpt primal solution back into a post-move State and move list."""
    after = deepcopy(state)
    after_by_id = {u["unit_id"]: u for u in after["units"]}
    moves: list[dict] = []
    for u in range(build.U):
        uid = build.avail_unit_ids[u]
        home = build.home_station[u]
        chosen = max(range(build.S), key=lambda s: x[build.xi(u, s)])
        after_by_id[uid]["station"] = chosen
        if chosen != home:
            eta = _reloc_min(world, home, chosen)
            moves.append({
                "unit_id": uid,
                "from": home,
                "to": chosen,
                "eta_min": round(eta, 1),
                "reason": _move_reason(chosen, world, coverage_before, eta, rc),
            })
    return after, moves


def _normalize_status(reason: str) -> str:
    r = (reason or "").lower()
    if "optimal" in r:
        return "Optimal"
    if "infeasible" in r:
        return "Infeasible"
    if "feasible" in r:
        return "Feasible"
    return "Error" if ("error" in r or not reason) else reason


def _avail_homes(state: C.State) -> list[int]:
    return [u["station"] for u in state["units"] if u["status"] == "available"]


def _reloc_need(state: C.State, world: C.World, rc: dict, fsas) -> tuple[Optional[float], set]:
    """For operator-commanded FSAs, the smallest relocation drive-time that lets
    SOME available unit reach SOME station covering that zone (within threshold).

    Returns (cap_needed, uncoverable):
      cap_needed  = max over coverable commanded zones of that smallest distance —
                    i.e. the max_reloc_min that would let the fleet reach all of them
                    (None if no commanded zone is coverable at all).
      uncoverable = commanded zones with NO covering station reachable at any distance.
    """
    homes = _avail_homes(state)
    st, tt, thr = world["station_travel"], world["travel_time"], rc["threshold_min"]
    fsa_to_z = {world["fsa_index"][z]: z for z in range(world["Z"])}
    needs, uncoverable = [], set()
    for fsa in fsas:
        z = fsa_to_z.get(fsa)
        if z is None:
            continue
        cov_st = [s for s in range(world["S"]) if float(tt[s, z]) <= thr]
        reach = [float(st[h, s]) for s in cov_st for h in homes
                 if float(st[h, s]) < C.UNREACHABLE_MIN / 2]
        if reach:
            needs.append(min(reach))
        else:
            uncoverable.add(fsa)
    return (max(needs) if needs else None), uncoverable


def _solve_once(state: C.State, world: C.World, rc: dict) -> C.OptimizeResult:
    """Build + solve the relocation MILP for one fully-resolved constraint set."""
    from cuopt.linear_programming import solver
    from cuopt.linear_programming.solver_settings import SolverSettings

    coverage_before = evaluate(state, world, rc["threshold_min"])
    build = build_milp(state, world, rc)
    if build.U == 0 or build.data_model is None:
        return {"moves": [], "coverage_before": coverage_before,
                "coverage_after": coverage_before, "objective": 0.0,
                "solve_time_ms": 0.0, "status": "Optimal", "n_moves": 0, "notes": []}

    settings = SolverSettings()
    settings.set_parameter("log_to_console", False)
    sol = solver.Solve(build.data_model, settings)
    status = _normalize_status(sol.get_termination_reason())
    solve_ms = float(sol.get_solve_time()) * 1000.0

    if status == "Infeasible":
        return {"moves": [], "coverage_before": coverage_before,
                "coverage_after": coverage_before, "objective": 0.0,
                "solve_time_ms": solve_ms, "status": "Infeasible", "n_moves": 0, "notes": []}

    after, moves = _decode(sol.get_primal_solution(), build, state, world, coverage_before, rc)
    return {"moves": moves, "coverage_before": coverage_before,
            "coverage_after": evaluate(after, world, rc["threshold_min"]),
            "objective": float(sol.get_primal_objective()), "solve_time_ms": solve_ms,
            "status": status, "n_moves": len(moves), "notes": []}


def re_optimize(
    state: C.State,
    world: C.World,
    constraints: Optional[C.Constraints] = None,
) -> C.OptimizeResult:
    """Solve the relocation MILP, honoring operator commands.

    Soft steering (zone_priority) is solved directly. For a HARD protect_zones
    guarantee that the default locality cap (max_reloc_min) can't satisfy, we
    ESCALATE rather than silently return a do-nothing result — which would regress
    coverage *below* what the operator would get with no command at all:
      1. relax max_reloc_min just enough to reach the commanded zone(s), re-solve;
      2. if a commanded zone has no covering station reachable at any distance, drop
         it from the hard guarantee, boost it strongly instead, and keep optimizing
         the rest of the fleet — attaching a `notes` entry explaining what happened.
    The result carries a `notes: list[str]` the loop/UI can show the operator.
    """
    rc = C.resolve_constraints(constraints, world)
    res = _solve_once(state, world, rc)

    protect = list(rc["protect_zones"])
    if not protect or res["status"] != "Infeasible":
        return res

    # A hard guarantee is infeasible at the current locality cap — escalate.
    need, uncoverable = _reloc_need(state, world, rc, protect)
    reachable = sorted(set(protect) - uncoverable)

    # 1) relax the locality cap to reach the commanded zones, then re-solve.
    if need is not None:
        cap = rc["max_reloc_min"]
        if cap is None or need + 0.5 > cap:
            rc = {**rc, "max_reloc_min": round(need + 0.5, 1)}
            r2 = _solve_once(state, world, rc)
            if r2["status"] != "Infeasible":
                r2["notes"] = [f"relaxed relocation cap to {rc['max_reloc_min']:.0f} min to "
                               f"honor commanded zone(s): {', '.join(reachable)}"]
                return r2
            res = r2

    # 2) still infeasible -> drop uncoverable zones to a strong boost, optimize rest.
    rc_fb = {**rc, "protect_zones": reachable,
             "zone_priority": {**rc["zone_priority"],
                               **{z: max(rc["zone_priority"].get(z, 1.0), 6.0) for z in uncoverable}}}
    r3 = _solve_once(state, world, rc_fb)
    notes = []
    if uncoverable:
        notes.append(f"{', '.join(sorted(uncoverable))}: no available unit can reach a covering "
                     f"post — boosted priority instead of guaranteeing coverage")
    if r3["status"] == "Infeasible":
        # nothing with a hard guarantee worked; best-effort with no protect.
        r3 = _solve_once(state, world, {**rc, "protect_zones": [],
                                        "zone_priority": rc_fb["zone_priority"]})
        notes.append("could not guarantee the commanded zone(s); optimized for best overall coverage")
    r3["notes"] = notes
    return r3


def _move_reason(to_station: int, world: C.World, before: C.CoverageMap,
                 eta_min: float, rc: Optional[dict] = None) -> str:
    """Explain a relocation, crediting the operator's command when it drove it.

    Priority of rationale (most operator-relevant first):
      1. a protected zone the destination now covers   -> protect:FSA
      2. a boosted zone (zone_priority>1) it covers     -> boost:FSA
      3. the highest-demand coverage gap it heals        -> cover_gap:FSA
      4. otherwise                                       -> rebalance
    The real relocation ETA is always appended so the tag is decision-grade.
    """
    rc = rc or {}
    protect = set(rc.get("protect_zones", []))
    zpri = rc.get("zone_priority", {})
    covers = lambda z: world["coverage"][to_station, z]

    what = None
    # 1. honoring a hard protect
    for z in range(world["Z"]):
        fsa = world["fsa_index"][z]
        if fsa in protect and covers(z):
            what = f"protect:{fsa}"
            break
    # 2. serving a soft boost (pick the most-boosted demand it reaches)
    if what is None and zpri:
        best_fsa, best_w = None, 0.0
        for z in range(world["Z"]):
            fsa = world["fsa_index"][z]
            mult = zpri.get(fsa, 1.0)
            if mult > 1.0 and covers(z):
                w = world["fsa_meta"][z]["demand_weight"] * mult
                if w > best_w:
                    best_w, best_fsa = w, fsa
        if best_fsa:
            what = f"boost:{best_fsa}"
    # 3. healing the biggest demand gap
    if what is None:
        gaps = set(before["gaps"])
        best_fsa, best_dem = None, -1.0
        for z in range(world["Z"]):
            fsa = world["fsa_index"][z]
            if fsa in gaps and covers(z):
                d = world["fsa_meta"][z]["demand_weight"]
                if d > best_dem:
                    best_dem, best_fsa = d, fsa
        what = f"cover_gap:{best_fsa}" if best_fsa else "rebalance"
    return f"{what} ({eta_min:.1f} min reloc)"


def re_optimize_robust(
    state: C.State,
    world: C.World,
    constraints: Optional[C.Constraints] = None,
    n_scenarios: int = 8,
    n_score: int = 500_000,
    concentration: float = 30.0,
    seed: int = 0,
) -> C.OptimizeResult:
    """Robust relocation under demand uncertainty — the GPU-filling optimizer.

    Instead of one deterministic solve, build a PORTFOLIO of MILPs — one for the
    historical demand plus `n_scenarios` sampled demand realizations — and solve
    them together with cuOpt BatchSolve (parallel on the GPU). Each solve yields
    a candidate placement; we then score all candidates (plus a do-nothing
    baseline) against `n_score` fresh demand scenarios with the Monte Carlo
    ranking engine and return the placement that is most robust on average.

    Returns the standard OptimizeResult shape (so Person 3's loop / LLM are
    unchanged), with an extra `robust` block describing the portfolio.
    """
    import numpy as np
    from cuopt.linear_programming.solver import BatchSolve
    from cuopt.linear_programming.solver_settings import SolverSettings

    from optimizer.montecarlo import _demand_base, rank_plans, sample_demand

    rc = C.resolve_constraints(constraints, world)
    coverage_before = evaluate(state, world, rc["threshold_min"])

    probe = build_milp(state, world, rc)
    if probe.U == 0 or probe.data_model is None:
        return {
            "moves": [], "coverage_before": coverage_before,
            "coverage_after": coverage_before, "objective": 0.0,
            "solve_time_ms": 0.0, "status": "Optimal", "n_moves": 0,
            "robust": {"chosen_plan": "do_nothing", "n_plans_scored": 1,
                       "n_scenarios_solved": 0, "batch_solve_ms": 0.0,
                       "ranking": []},
        }

    # demand portfolio: historical + sampled scenarios
    samp = sample_demand(world, n_scenarios, concentration, seed)
    try:
        import cupy as cp
        samp = cp.asnumpy(samp)
    except Exception:
        samp = np.asarray(samp)

    overrides = [None] + [samp[i] for i in range(n_scenarios)]   # None => historical
    sc_labels = ["deterministic"] + [f"scenario_{i + 1}" for i in range(n_scenarios)]
    builds = [build_milp(state, world, rc, demand_override=o) for o in overrides]

    settings = SolverSettings()
    settings.set_parameter("log_to_console", False)
    sols, batch_s = BatchSolve([b.data_model for b in builds], settings)
    batch_ms = float(batch_s) * 1000.0

    # candidate placements: do-nothing baseline + every feasible solve
    plans: list[C.State] = [state]
    plan_labels: list[str] = ["do_nothing"]
    plan_moves: list[list[dict]] = [[]]
    for i, sol in enumerate(sols):
        if _normalize_status(sol.get_termination_reason()) == "Infeasible":
            continue
        after, moves = _decode(sol.get_primal_solution(), builds[i],
                               state, world, coverage_before, rc)
        plans.append(after); plan_labels.append(sc_labels[i]); plan_moves.append(moves)

    # rank by robust expected coverage over FRESH scenarios (seed+1) to avoid
    # rewarding a plan just for fitting the draws it was optimized on.
    # NOTE: operator zone_priority/forbid steer candidate GENERATION (each build_milp
    # sees the reweighted demand), but this ranking scores on base historical demand,
    # so a low-demand boosted zone may not win the final pick here. The loop uses the
    # simple re_optimize() above, which honors zone_priority directly; to also honor it
    # in the robust pick, weight rank_plans/sample_demand by zone_priority (TODO).
    rk = rank_plans(plans, world, n_scenarios=n_score, concentration=concentration,
                    threshold_min=rc["threshold_min"], seed=seed + 1, labels=plan_labels)
    best = rk["best"]
    bi = best["index"]
    after = plans[bi]
    moves = plan_moves[bi]
    coverage_after = evaluate(after, world, rc["threshold_min"])

    return {
        "moves": moves,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "objective": float(best["expected_coverage_pct"]),
        "solve_time_ms": batch_ms,
        "status": "Optimal",
        "n_moves": len(moves),
        "robust": {
            "chosen_plan": plan_labels[bi],
            "expected_coverage_pct": float(best["expected_coverage_pct"]),
            "p5": float(best["p5"]), "p95": float(best["p95"]),
            "n_plans_scored": len(plans),
            "n_scenarios_solved": n_scenarios,
            "score_scenarios": int(n_score),
            "batch_solve_ms": batch_ms,
            "ranking": rk["ranking"][:5],
        },
    }
