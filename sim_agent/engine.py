"""
Discrete-event simulation engine — Person 3.

Generates the live dynamics the whole demo runs on: emergency calls arrive (a
Poisson process whose spatial distribution is the FSA demand_weight), the
nearest available unit is dispatched and goes busy for a service period, then
returns to its post. At any moment the sim can emit a `State` (CONTRACTS.md §2)
that Person 2's `evaluate` / `re_optimize` consume.

Pure-python + numpy; no GPU. The matrix and coverage live in `world`, never here.
"""
from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

import contracts as C

CALL_ARRIVAL = "call_arrival"
UNIT_FREE = "unit_free"


@dataclass(order=True)
class _Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


class Simulation:
    """Event-driven ambulance sim producing contract-shaped `State` snapshots.

    Parameters
    ----------
    world : C.World            the static graph/coverage Person 1 ships
    n_units : int              number of ambulances in the system
    calls_per_hour : float     system-wide arrival rate (Poisson)
    mean_service_min : float   mean on-scene + transport time per call (exp.)
    seed : int                 RNG seed for reproducible demos
    """

    def __init__(
        self,
        world: C.World,
        n_units: int = 12,
        calls_per_hour: float = 18.0,
        mean_service_min: float = 40.0,
        seed: int = 0,
    ):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self._seq = 0
        self._heap: list[_Event] = []

        self.rate_per_min = calls_per_hour / 60.0
        self.mean_service = mean_service_min
        self.threshold = float(world["threshold_min"])

        self.demand = np.array(
            [world["fsa_meta"][z]["demand_weight"] for z in range(world["Z"])]
        )
        self.demand = self.demand / self.demand.sum()

        self.units = self._init_units(n_units)

        # running KPIs across the whole run
        self.stats = {
            "calls": 0,
            "served": 0,
            "missed": 0,            # no available unit at all
            "on_time": 0,           # response <= threshold
            "late": 0,              # served but response > threshold
            "response_times": [],   # minutes, served calls only
        }

        self._schedule_next_call()

    # ----------------------------------------------------------------- setup
    def _station_score(self) -> np.ndarray:
        """Per-station value = total historical demand it can cover within threshold.

        Vector (S,). Stations that reach lots of busy zones score high; remote
        stations covering only quiet zones score low. Drives initial placement.
        """
        cov = self.world["coverage"].astype("float64")          # (S, Z)
        return cov @ self.demand                                # (S,)

    def _init_units(self, n_units: int) -> list[C.Unit]:
        """Place units by historical demand, not blindly round-robin.

        Greedy load-balanced fill: each unit goes to the station maximizing
        score / (1 + units_already_there). High-demand stations therefore attract
        more units, but the diminishing 1/(1+k) term keeps clustering in check, so
        quiet stations still get a unit or two — exactly the low-priority coverage a
        dispatcher command can later pull away to reinforce a hotspot.

        Deterministic given the world (ties broken by lowest s-index), so demos
        reproduce. Falls back to round-robin only if scores are all zero.
        """
        S = self.world["S"]
        score = self._station_score()
        if not np.any(score > 0):                               # degenerate world
            homes = [i % S for i in range(n_units)]
        else:
            counts = np.zeros(S, dtype=int)
            homes = []
            for _ in range(n_units):
                s = int(np.argmax(score / (1.0 + counts)))
                homes.append(s)
                counts[s] += 1
        return [
            {
                "unit_id": f"AMB_{i:02d}",
                "status": "available",
                "station": homes[i],
                "busy_until": None,
            }
            for i in range(n_units)
        ]

    # --------------------------------------------------------------- helpers
    def _push(self, time: float, kind: str, payload: dict | None = None) -> None:
        self._seq += 1
        heapq.heappush(self._heap, _Event(time, self._seq, kind, payload or {}))

    def _schedule_next_call(self) -> None:
        gap = float(self.rng.exponential(1.0 / self.rate_per_min))
        self._push(self.t + gap, CALL_ARRIVAL)

    def _sample_zone(self) -> int:
        return int(self.rng.choice(self.world["Z"], p=self.demand))

    def _nearest_available(self, z: int) -> tuple[int | None, float]:
        """(unit list-index, travel_time) of closest available unit to zone z."""
        tt = self.world["travel_time"]
        best_i, best_t = None, C.UNREACHABLE_MIN
        for i, u in enumerate(self.units):
            if u["status"] != "available":
                continue
            drive = float(tt[u["station"], z])
            if drive < best_t:
                best_i, best_t = i, drive
        return best_i, best_t

    # ----------------------------------------------------------- event logic
    def _handle_call(self, _payload: dict) -> dict:
        self.stats["calls"] += 1
        z = self._sample_zone()
        i, drive = self._nearest_available(z)

        record = {"t": self.t, "zone": self.world["fsa_index"][z], "served": False}

        if i is None or drive >= C.UNREACHABLE_MIN:
            self.stats["missed"] += 1
        else:
            unit = self.units[i]
            service = float(self.rng.exponential(self.mean_service))
            # busy = drive to scene + on-scene/transport service time
            unit["status"] = "busy"
            unit["busy_until"] = self.t + drive + service
            self._push(unit["busy_until"], UNIT_FREE, {"unit_index": i})

            self.stats["served"] += 1
            self.stats["response_times"].append(drive)
            if drive <= self.threshold:
                self.stats["on_time"] += 1
            else:
                self.stats["late"] += 1
            record.update(served=True, unit=unit["unit_id"], response_min=drive)

        self._schedule_next_call()
        return record

    def _handle_free(self, payload: dict) -> None:
        unit = self.units[payload["unit_index"]]
        unit["status"] = "available"
        unit["busy_until"] = None  # returns to (possibly relocated) post

    # ------------------------------------------------------------------- API
    def run_until(self, t_target: float) -> list[dict]:
        """Advance the clock to t_target, processing every event in between.

        Returns the list of call records that occurred (handy for logging).
        """
        call_log: list[dict] = []
        while self._heap and self._heap[0].time <= t_target:
            ev = heapq.heappop(self._heap)
            self.t = ev.time
            if ev.kind == CALL_ARRIVAL:
                call_log.append(self._handle_call(ev.payload))
            elif ev.kind == UNIT_FREE:
                self._handle_free(ev.payload)
        self.t = max(self.t, t_target)
        return call_log

    def snapshot(self) -> C.State:
        """Current contract-shaped State (deep-copied so callers can't mutate)."""
        return {"t": self.t, "units": deepcopy(self.units)}

    def apply_moves(self, moves: list[dict]) -> list[dict]:
        """Relocate available units per an OptimizeResult's `moves`.

        Busy units are never relocated mid-call; such a move is skipped and
        flagged in the returned audit list.
        """
        by_id = {u["unit_id"]: u for u in self.units}
        applied = []
        for m in moves:
            u = by_id.get(m["unit_id"])
            if u is None:
                applied.append({**m, "applied": False, "why": "unknown unit"})
            elif u["status"] != "available":
                applied.append({**m, "applied": False, "why": "unit busy"})
            else:
                u["station"] = m["to"]
                applied.append({**m, "applied": True})
        return applied

    def kpis(self) -> dict:
        s = self.stats
        rt = np.array(s["response_times"]) if s["response_times"] else np.array([0.0])
        served = max(s["served"], 1)
        return {
            "calls": s["calls"],
            "served": s["served"],
            "missed": s["missed"],
            "on_time_pct": s["on_time"] / served,
            "mean_response_min": float(rt.mean()),
            "p90_response_min": float(np.percentile(rt, 90)),
        }
