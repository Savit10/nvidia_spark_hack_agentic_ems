"""
Coverage evaluator — Person 2.

Pure, fast, non-cuOpt. Given the current State (who's available and where) plus
the static World, returns a CoverageMap: which demand zones are covered, the
demand-weighted coverage %, and the per-zone detail the map render needs.

Default policy (CONTRACTS.md §2): only `available` units provide coverage;
busy units do not.
"""
from __future__ import annotations

import numpy as np

import contracts as C


def available_stations(state: C.State) -> list[int]:
    """s-indices currently staffed by an available unit."""
    return [u["station"] for u in state["units"] if u["status"] == "available"]


def covered_mask(state: C.State, world: C.World, threshold_min: float | None = None) -> np.ndarray:
    """
    Boolean vector (Z,): is each zone covered by >=1 available unit within threshold?

    Uses world["coverage"] when threshold matches world's, else recomputes from
    travel_time (so an LLM threshold override works without rebuilding World).
    """
    stations = available_stations(state)
    Z = world["Z"]
    if not stations:
        return np.zeros(Z, dtype=bool)

    tt = world["travel_time"][stations, :]                  # (k, Z)
    if threshold_min is None or threshold_min == world["threshold_min"]:
        cov = world["coverage"][stations, :]                # (k, Z)
    else:
        cov = tt <= threshold_min
    return cov.any(axis=0)                                  # (Z,)


def nearest_unit_minutes(state: C.State, world: C.World) -> np.ndarray:
    """Vector (Z,): travel time of the closest available unit to each zone."""
    stations = available_stations(state)
    Z = world["Z"]
    if not stations:
        return np.full(Z, C.UNREACHABLE_MIN, dtype="float32")
    return world["travel_time"][stations, :].min(axis=0)    # (Z,)


def evaluate(state: C.State, world: C.World, threshold_min: float | None = None) -> C.CoverageMap:
    mask = covered_mask(state, world, threshold_min)
    nearest = nearest_unit_minutes(state, world)
    demand = np.array([world["fsa_meta"][z]["demand_weight"] for z in range(world["Z"])])

    covered, gaps, per_zone = [], [], {}
    for z in range(world["Z"]):
        fsa = world["fsa_index"][z]
        is_cov = bool(mask[z])
        (covered if is_cov else gaps).append(fsa)
        per_zone[fsa] = {
            "covered": is_cov,
            "nearest_unit_min": float(nearest[z]),
            "demand_weight": float(demand[z]),
        }

    covered_demand_pct = float((demand * mask).sum())       # demand normalized to 1
    return {
        "covered_demand_pct": covered_demand_pct,
        "covered": covered,
        "gaps": gaps,
        "per_zone": per_zone,
    }
