"""
Pre-record a simulation timeline for the Gotham-style web frontend.

Runs the REAL integrated loop (sim -> evaluate -> cuOpt re_optimize -> agent
explanation) for N decision ticks and dumps a single JSON the static frontend
plays back. No live backend needed at demo time.

Run from repo root in the cuOpt env:
    source ~/cuopt-env/bin/activate
    python -m webapp.generate_timeline
    # nicer narration (slower): NIM_BASE_URL=http://localhost:11434/v1 NIM_MODEL=nemotron3:33b python -m webapp.generate_timeline

Outputs:
    webapp/data/world.json      stations, threshold, bounds, demand per FSA
    webapp/data/timeline.json   per-tick units / coverage / moves / KPIs / text
    webapp/data/toronto_fsa.geojson  (already copied)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import contracts as C  # noqa: E402
from load_world import load_world  # noqa: E402
from optimizer import evaluate  # noqa: E402
from sim_agent import agent  # noqa: E402
from sim_agent.engine import Simulation  # noqa: E402
from sim_agent.loop import Loop  # noqa: E402

DATA = os.path.join(ROOT, "webapp", "data")


def station_lonlat(world, s):
    m = world["station_meta"][s]
    return [m["lon"], m["lat"]]


def build_world_json(world):
    stations = [
        {"s": s, "id": world["station_meta"][s]["station_id"],
         "name": world["station_meta"][s]["name"],
         "lat": world["station_meta"][s]["lat"], "lon": world["station_meta"][s]["lon"]}
        for s in range(world["S"])
    ]
    lats = [s["lat"] for s in stations]
    lons = [s["lon"] for s in stations]
    # FSAs no station can reach within threshold AT ALL (coverage column empty) —
    # structurally uncoverable, distinct from a transient gap. Expect [] after the
    # build's reachable-node repair; kept as a live safety net + regression check.
    cov = world["coverage"]
    uncoverable = [world["fsa_index"][z] for z in range(world["Z"])
                   if not bool(cov[:, z].any())]
    return {
        "threshold_min": world["threshold_min"],
        "S": world["S"], "Z": world["Z"],
        "stations": stations,
        "fsa_demand": {world["fsa_index"][z]: world["fsa_meta"][z]["demand_weight"]
                       for z in range(world["Z"])},
        "fsa_uncoverable": uncoverable,
        "center": {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)},
        "bounds": {"min_lat": min(lats), "max_lat": max(lats),
                   "min_lon": min(lons), "max_lon": max(lons)},
    }


def tick_record(world, loop, tr, scene="NORMAL"):
    state = loop.sim.snapshot()
    # recompute live coverage on the post-step state so the map is accurate
    cov = evaluate(state, world, world["threshold_min"])

    units = [
        {"id": u["unit_id"],
         "lat": world["station_meta"][u["station"]]["lat"],
         "lon": world["station_meta"][u["station"]]["lon"],
         "status": u["status"],
         # roster-card detail: which post it's at + minutes until it frees (busy)
         "post": world["station_meta"][u["station"]]["name"],
         "station_id": world["station_meta"][u["station"]].get("station_id", ""),
         "eta_free": (max(0, round(u["busy_until"] - state["t"]))
                      if u.get("busy_until") else None)}
        for u in state["units"]
    ]
    coverage = {
        fsa: {"covered": pz["covered"], "demand": pz["demand_weight"],
              "nearest_min": round(pz["nearest_unit_min"], 1)}
        for fsa, pz in cov["per_zone"].items()
    }
    moves = [
        {"unit": m["unit_id"],
         "from": station_lonlat(world, m["from"]),
         "to": station_lonlat(world, m["to"]),
         "eta": m["eta_min"], "reason": m.get("reason", "")}
        for m in (tr.applied or [])
        if m.get("applied", True)
    ]
    # enriched 911 calls this tick -> incident feed + map blips
    incidents = [
        {"t": round(c["t"], 1), "label": c.get("label", ""),
         "type": c.get("incident_type", ""), "priority": c.get("priority_label", ""),
         "zone": c["zone"], "served": c["served"], "unit": c.get("unit"),
         "resp": round(c["response_min"], 1) if c.get("served") else None,
         "lat": c.get("lat"), "lon": c.get("lon")}
        for c in (tr.calls or []) if "lat" in c
    ]
    # dispatch trips (unit drives station -> scene -> back), real event timing
    trips = [
        {"unit": c["unit"], "from": [c["from_lon"], c["from_lat"]],
         "to": [c["lon"], c["lat"]], "depart": round(c["t"], 2),
         "arrive": round(c["t"] + c["response_min"], 2),
         "free": c.get("busy_until", round(c["t"] + c["response_min"], 2))}
        for c in (tr.calls or []) if c.get("served") and "from_lat" in c
    ]
    solve = None
    if tr.result is not None:
        r = tr.result
        solve = {"solve_time_ms": round(r["solve_time_ms"], 2), "status": r["status"],
                 "n_moves": r["n_moves"], "objective": round(r["objective"], 4)}
    kp = loop.sim.kpis()
    return {
        "t": round(tr.t, 1),
        "scene": scene,
        "calls_this_tick": tr.calls_this_tick,
        "units": units,
        "coverage": coverage,
        "covered_pct": round(cov["covered_demand_pct"], 4),
        "gaps": len(cov["gaps"]),
        "moves": moves,
        "trips": trips,
        "incidents": incidents,
        "solve": solve,
        "kpi": {"calls": kp["calls"], "served": kp["served"], "missed": kp["missed"],
                "on_time_pct": round(kp["on_time_pct"], 3),
                "mean_resp": round(kp["mean_response_min"], 1)},
        "explanation": tr.explanation,
        "command": tr.command,
        "reasoning": (tr.result or {}).get("reasoning"),
        "notes": (tr.result or {}).get("notes", []),
        "decision": (tr.result or {}).get("decision"),
        "intent": getattr(tr, "intent", "coverage"),
        "dispatched": getattr(tr, "dispatched", None),
        "timings": getattr(tr, "timings", {}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    # defaults tuned for a clean cinematic arc: stable ~97% normal ops, a surge
    # that visibly cracks coverage to ~48%, then a cuOpt heal back to ~94%.
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--tick-min", type=float, default=10.0)
    ap.add_argument("--units", type=int, default=17)
    ap.add_argument("--calls-per-hour", type=float, default=6.0)
    ap.add_argument("--service", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--surge-ticks", type=int, default=4, help="length of the 911 surge")
    ap.add_argument("--surge-mult", type=float, default=7.0, help="arrival-rate x during surge")
    args = ap.parse_args(argv)

    world = load_world(os.path.join(ROOT, "datasets", "world.npz"),
                       os.path.join(ROOT, "datasets", "world_meta.json"))

    print(f"NIM agent: {'online (Nemotron)' if agent.nim_available() else 'offline -> rule/template fallback'}")
    profiles = os.path.join(ROOT, "datasets", "incident_profiles.json")
    profiles = profiles if os.path.exists(profiles) else None
    sim = Simulation(world, n_units=args.units, calls_per_hour=args.calls_per_hour,
                     mean_service_min=args.service, seed=args.seed,
                     incident_profiles=profiles)
    loop = Loop(sim, world, tick_min=args.tick_min)

    # --- scripted narrative arc -------------------------------------------- #
    # A 911 SURGE (calls spike ~3x) cracks coverage open; then the dispatcher
    # keys the radio and cuOpt heals it. surge -> radio/solve -> recovered.
    base_rate = sim.rate_per_min
    surge_start = max(2, int(args.ticks * 0.45))
    surge_len = max(2, args.surge_ticks)
    heal_tick = surge_start + surge_len            # operator command + cuOpt heal

    def scene_for(k):
        if surge_start <= k < heal_tick:
            return "SURGE"
        if k == heal_tick:
            return "SOLVE"
        if heal_tick < k <= heal_tick + 2:
            return "RECOVERED"
        return "NORMAL"

    ticks = []
    for k in range(args.ticks):
        scene = scene_for(k)
        # drive the surge by temporarily multiplying the arrival rate
        sim.rate_per_min = base_rate * (args.surge_mult if scene == "SURGE" else 1.0)
        # PAUSE relocation during the surge so coverage visibly cracks open; the
        # operator's radio call at the SOLVE tick is what fires cuOpt to heal it.
        loop.optimize = (scene != "SURGE")

        cmd = None
        if k == heal_tick:
            cov_now = evaluate(loop.sim.snapshot(), world, world["threshold_min"])
            gaps_by_demand = sorted(
                cov_now["gaps"],
                key=lambda f: cov_now["per_zone"][f]["demand_weight"], reverse=True)
            if gaps_by_demand:
                cmd = (f"Make sure {gaps_by_demand[0]} stays covered and "
                       f"don't make more than 3 moves.")
                print(f"  [tick {k}] operator command -> cover {gaps_by_demand[0]}")
        tr = loop.step(command=cmd)
        ticks.append(tick_record(world, loop, tr, scene=scene_for(k)))
        t = ticks[-1]
        print(f"  tick {k:2d}  {t['scene']:9s} t={tr.t:5.0f}  cov={t['covered_pct']*100:5.1f}%  "
              f"gaps={t['gaps']:2d}  calls={t['calls_this_tick']:2d}  moves={len(t['moves'])}")

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "world.json"), "w") as f:
        json.dump(build_world_json(world), f)
    timeline = {
        "meta": {"threshold_min": world["threshold_min"], "tick_min": args.tick_min,
                 "n_ticks": len(ticks), "units": args.units},
        "ticks": ticks,
    }
    with open(os.path.join(DATA, "timeline.json"), "w") as f:
        json.dump(timeline, f)

    print(f"\nwrote {len(ticks)} ticks -> webapp/data/timeline.json + world.json")


if __name__ == "__main__":
    main()
