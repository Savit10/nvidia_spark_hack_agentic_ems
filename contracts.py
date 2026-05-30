"""
Shared interface contract for the ambulance-relocation project.

This module is the FROZEN boundary between the three lanes (see CONTRACTS.md).
It depends only on numpy so every lane can import it regardless of its CUDA env.

Person 1 (data) produces `World`.
Person 2 (optimizer) consumes World + State (+Constraints) -> OptimizeResult.
Person 3 (sim + LLM) drives the loop and produces State + Constraints.
"""
from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

# --------------------------------------------------------------------------- #
#  Typed shapes (field names ARE the contract — see CONTRACTS.md §1-§5)
# --------------------------------------------------------------------------- #


class FsaMeta(TypedDict):
    fsa: str
    lat: float
    lon: float
    demand_weight: float


class StationMeta(TypedDict):
    station_id: str
    name: str
    lat: float
    lon: float
    capacity: int


class World(TypedDict):
    fsa_index: list[str]            # len Z, position = z-index
    station_index: list[str]        # len S, position = s-index
    fsa_meta: dict[int, FsaMeta]
    station_meta: dict[int, StationMeta]
    travel_time: np.ndarray         # (S, Z) float32 minutes; unreachable = BIG, never inf/nan
    coverage: np.ndarray            # (S, Z) bool == (travel_time <= threshold_min)
    station_travel: np.ndarray      # (S, S) float32 minutes station->station, diagonal 0
    threshold_min: float
    Z: int
    S: int


class Unit(TypedDict):
    unit_id: str
    status: str                     # "available" | "busy"
    station: int                    # s-index
    busy_until: Optional[float]


class State(TypedDict):
    t: float
    units: list[Unit]


class Constraints(TypedDict, total=False):
    lock_units: list[str]
    force_station: dict[str, int]
    max_moves: Optional[int]
    protect_zones: list[str]
    threshold_min: Optional[float]
    move_penalty: Optional[float]
    max_reloc_min: Optional[float]     # hard cap: a unit may not relocate farther than this (drive min)
    familiar_min: Optional[float]      # familiarity scale: a move of this length costs one move_penalty unit


class Move(TypedDict):
    unit_id: str
    from_: int                      # NOTE: 'from' is reserved; serialize as "from"
    to: int
    eta_min: float
    reason: str


class PerZone(TypedDict):
    covered: bool
    nearest_unit_min: float
    demand_weight: float


class CoverageMap(TypedDict):
    covered_demand_pct: float
    covered: list[str]
    gaps: list[str]
    per_zone: dict[str, PerZone]


class OptimizeResult(TypedDict):
    moves: list[dict]               # {"unit_id","from","to","eta_min","reason"}
    coverage_before: CoverageMap
    coverage_after: CoverageMap
    objective: float
    solve_time_ms: float
    status: str                     # "Optimal" | "Feasible" | "Infeasible" | "Error"
    n_moves: int


# Sentinel for an unreachable cell in travel_time (finite, large).
UNREACHABLE_MIN: float = 1.0e6


# --------------------------------------------------------------------------- #
#  Default constraint values (CONTRACTS.md §3)
# --------------------------------------------------------------------------- #

# move_penalty is in MULTIPLES OF MEAN-ZONE DEMAND (scale-invariant; see milp.py).
# 0.5 => a relocation of FAMILIAR_MIN minutes must net at least half an average
# zone's demand to happen. Cost now scales with the real relocation drive time.
DEFAULT_MOVE_PENALTY = 0.5

# A unit will not be repositioned farther than this many drive-minutes from its
# current post — dispatchers reposition locally, not across the city.
DEFAULT_MAX_RELOC_MIN = 10.0

# "Familiar territory" scale: a relocation of this length costs exactly one
# move_penalty unit; shorter moves are cheaper, longer ones ramp up linearly.
# Doubles as a route-familiarity proxy (nearby posts = familiar turf).
DEFAULT_FAMILIAR_MIN = 5.0


def resolve_constraints(c: Optional[Constraints], world: World) -> dict:
    """Fill missing fields with defaults so the optimizer never branches on None."""
    c = c or {}
    return {
        "lock_units": list(c.get("lock_units", [])),
        "force_station": dict(c.get("force_station", {})),
        "max_moves": c.get("max_moves", None),
        "protect_zones": list(c.get("protect_zones", [])),
        "threshold_min": c.get("threshold_min") or world["threshold_min"],
        "move_penalty": c.get("move_penalty") if c.get("move_penalty") is not None
        else DEFAULT_MOVE_PENALTY,
        "max_reloc_min": c.get("max_reloc_min") if c.get("max_reloc_min") is not None
        else DEFAULT_MAX_RELOC_MIN,
        "familiar_min": c.get("familiar_min") if c.get("familiar_min") is not None
        else DEFAULT_FAMILIAR_MIN,
    }


# --------------------------------------------------------------------------- #
#  Mocks (CONTRACTS.md §8) — let every lane build in parallel from minute one.
# --------------------------------------------------------------------------- #


def mock_world(S: int = 5, Z: int = 8, seed_offset: int = 0) -> World:
    rng = np.random.default_rng(42 + seed_offset)
    tt = rng.uniform(2, 20, size=(S, Z)).astype("float32")
    thr = 9.0
    dem = rng.uniform(0.05, 1.0, size=Z)
    dem /= dem.sum()
    # symmetric station->station drive times (min), diagonal 0
    st = rng.uniform(2, 25, size=(S, S)).astype("float32")
    st = ((st + st.T) / 2).astype("float32")
    np.fill_diagonal(st, 0.0)
    return {
        "fsa_index": [f"M{i}A" for i in range(Z)],
        "station_index": [f"STN_{i:03d}" for i in range(S)],
        "fsa_meta": {
            z: {"fsa": f"M{z}A", "lat": 43.6 + 0.01 * z, "lon": -79.4 - 0.01 * z,
                "demand_weight": float(dem[z])}
            for z in range(Z)
        },
        "station_meta": {
            s: {"station_id": f"STN_{s:03d}", "name": f"Station {s}",
                "lat": 43.65 + 0.01 * s, "lon": -79.38 - 0.01 * s, "capacity": 4}
            for s in range(S)
        },
        "travel_time": tt,
        "coverage": tt <= thr,
        "station_travel": st,
        "threshold_min": thr,
        "Z": Z,
        "S": S,
    }


def mock_state() -> State:
    return {
        "t": 120.0,
        "units": [
            {"unit_id": "AMB_00", "status": "available", "station": 0, "busy_until": None},
            {"unit_id": "AMB_01", "status": "available", "station": 1, "busy_until": None},
            {"unit_id": "AMB_02", "status": "busy", "station": 2, "busy_until": 145.0},
        ],
    }


def mock_constraints() -> Constraints:
    return {
        "lock_units": ["AMB_00"],
        "force_station": {},
        "max_moves": 2,
        "protect_zones": ["M3A"],
        "threshold_min": None,
        "move_penalty": 0.1,
    }


def mock_result() -> OptimizeResult:
    return {
        "moves": [{"unit_id": "AMB_01", "from": 1, "to": 3, "eta_min": 6.2,
                   "reason": "cover_gap:M3A"}],
        "coverage_before": {"covered_demand_pct": 0.71, "covered": ["M0A", "M1A"],
                            "gaps": ["M3A", "M6A"], "per_zone": {}},
        "coverage_after": {"covered_demand_pct": 0.89, "covered": ["M0A", "M1A", "M3A"],
                           "gaps": ["M6A"], "per_zone": {}},
        "objective": 0.89, "solve_time_ms": 14.3, "status": "Optimal", "n_moves": 1,
    }
