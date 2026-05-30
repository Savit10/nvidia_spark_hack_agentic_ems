"""
Relocation MILP — Person 2.

Builds the integer program that decides where each AVAILABLE unit should be
posted to maximize demand-weighted coverage, minus a penalty per relocation,
subject to the operator's constraints. Solves it with NVIDIA cuOpt on the GPU.

Model (CONTRACTS.md §7)
-----------------------
Variables
    x[u,s] in {0,1}   unit u posted at station s         index = u*S + s
    y[z]   in {0,1}    zone z is covered                  index = U*S + z

Maximize
    sum_z demand[z]*y[z]  -  move_penalty * sum_{u, s != home[u]} x[u,s]

Subject to
    (A) sum_s x[u,s] = 1                  for every available unit u
    (B) y[z] - sum_{u,s : cov[s,z]} x[u,s] <= 0     coverage linking
    (C) sum_u x[u,s] <= capacity[s]       station capacity
    (D) sum_{u, s != home[u]} x[u,s] <= max_moves   (only if max_moves set)

    protect_zones -> y[z] fixed to 1   (via bounds)
    lock_units / force_station -> x[u,s] fixed   (via bounds)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix

import contracts as C


@dataclass
class MilpBuild:
    """Everything needed to solve and then decode the solution."""
    data_model: object                       # cuopt DataModel
    avail_unit_ids: list[str]                # u-index -> unit_id
    home_station: list[int]                  # u-index -> current s-index
    S: int
    U: int
    Z: int
    n_vars: int
    threshold_min: float
    move_penalty: float
    notes: list[str] = field(default_factory=list)

    def xi(self, u: int, s: int) -> int:
        return u * self.S + s

    def yi(self, z: int) -> int:
        return self.U * self.S + z


def build_milp(state: C.State, world: C.World, constraints: dict,
               demand_override: "np.ndarray | None" = None) -> MilpBuild:
    """Construct the cuOpt DataModel. `constraints` must be resolved (no Nones).

    `demand_override` (len Z) replaces the historical demand weights in the
    objective — used to build a portfolio of MILPs over sampled demand scenarios
    for robust optimization (see reoptimize.re_optimize_robust)."""
    from cuopt.linear_programming.data_model import DataModel

    S, Z = world["S"], world["Z"]
    thr = constraints["threshold_min"]
    move_penalty = constraints["move_penalty"]

    avail = [u for u in state["units"] if u["status"] == "available"]
    avail_unit_ids = [u["unit_id"] for u in avail]
    home_station = [u["station"] for u in avail]
    U = len(avail)

    build = MilpBuild(
        data_model=None, avail_unit_ids=avail_unit_ids, home_station=home_station,
        S=S, U=U, Z=Z, n_vars=U * S + Z, threshold_min=thr, move_penalty=move_penalty,
    )
    if U == 0:
        build.notes.append("no available units; nothing to optimize")
        return build

    n_vars = build.n_vars

    # coverage[s,z] honoring a possible threshold override
    if thr == world["threshold_min"]:
        cov = world["coverage"]
    else:
        cov = world["travel_time"] <= thr

    # real graph drive-time between posting points (minutes), diagonal 0.
    st_travel = world["station_travel"]
    familiar_min = max(float(constraints["familiar_min"]), 1e-6)

    US = U * S
    home_arr = np.asarray(home_station, dtype=np.int64)             # (U,)
    # drive time home(u) -> s for every available unit (U, S)
    reloc_uS = np.asarray(st_travel)[home_arr, :].astype("float64")

    # ---- objective (maximize) — vectorized ------------------------------ #
    # move_penalty is expressed in MULTIPLES OF MEAN-ZONE DEMAND so it stays
    # scale-invariant: demand sums to 1 over Z zones, so a single avg zone is
    # worth 1/Z (~0.01 for 96 FSAs). A raw penalty would otherwise dwarf the
    # coverage reward and suppress all moves.
    #
    # The move cost is DISTANCE-PROPORTIONAL: a relocation of `familiar_min`
    # minutes costs exactly `move_penalty` mean-zones; a longer haul costs
    # linearly more. So the optimizer prefers short, familiar relocations and
    # only reaches farther when the coverage payoff clearly justifies it.
    # penalty=0.5 => "a familiar-distance move must net >= half an avg zone".
    mean_dem = 1.0 / Z
    penalty_coef = move_penalty * mean_dem
    reloc_cap = float(constraints["max_reloc_min"]) if constraints["max_reloc_min"] else 1e9

    obj = np.zeros(n_vars, dtype="float64")
    if demand_override is not None:
        obj[US:] = np.asarray(demand_override, dtype="float64")            # y reward
    else:
        obj[US:] = [world["fsa_meta"][z]["demand_weight"] for z in range(Z)]
    # x penalty: clip to the cap for a bounded coefficient; far/unreachable moves
    # are forbidden by bounds below, so the clip never hides them. Home move free.
    obj_x = -penalty_coef * (np.minimum(reloc_uS, reloc_cap) / familiar_min)   # (U,S)
    obj_x[np.arange(U), home_arr] = 0.0
    obj[:US] = obj_x.reshape(-1)

    # ---- constraint rows (COO blocks) — vectorized ---------------------- #
    # x[u,s] flattens to column u*S+s; y[z] to column U*S+z. Row ordering matches
    # the original: (A) U rows, (B) Z rows, (C) S rows, (D) optional 1 row.
    ri, rj, vv = [], [], []
    rhs_parts, rtype_parts = [], []
    r = 0

    # (A) each available unit assigned to exactly one station  (sum_s x[u,s] = 1)
    ri.append(np.repeat(np.arange(U), S))
    rj.append(np.arange(US))
    vv.append(np.ones(US))
    rhs_parts.append(np.ones(U)); rtype_parts.append(np.full(U, "E")); r += U

    # (B) coverage link: y[z] - sum_{u,s:cov[s,z]} x[u,s] <= 0
    ri.append(r + np.arange(Z)); rj.append(US + np.arange(Z)); vv.append(np.ones(Z))
    s_cov, z_cov = np.nonzero(cov)                       # cov is (S, Z)
    ncov = s_cov.shape[0]
    if ncov:
        ri.append(r + np.tile(z_cov, U))
        rj.append(np.repeat(np.arange(U), ncov) * S + np.tile(s_cov, U))
        vv.append(np.full(U * ncov, -1.0))
    rhs_parts.append(np.zeros(Z)); rtype_parts.append(np.full(Z, "L")); r += Z

    # (C) station capacity: sum_u x[u,s] <= capacity[s]
    ri.append(r + np.repeat(np.arange(S), U))
    rj.append(np.tile(np.arange(U), S) * S + np.repeat(np.arange(S), U))
    vv.append(np.ones(S * U))
    caps = np.array([float(world["station_meta"][s].get("capacity", U)) for s in range(S)])
    rhs_parts.append(caps); rtype_parts.append(np.full(S, "L")); r += S

    # (D) max_moves: sum_{u, s != home[u]} x[u,s] <= max_moves
    if constraints["max_moves"] is not None:
        u_grid = np.repeat(np.arange(U), S)
        s_grid = np.tile(np.arange(S), U)
        mv = s_grid != home_arr[u_grid]
        cols_d = (u_grid * S + s_grid)[mv]
        ri.append(np.full(cols_d.shape[0], r)); rj.append(cols_d)
        vv.append(np.ones(cols_d.shape[0]))
        rhs_parts.append(np.array([float(constraints["max_moves"])]))
        rtype_parts.append(np.array(["L"])); r += 1

    rows_i = np.concatenate(ri); rows_j = np.concatenate(rj); vals = np.concatenate(vv)
    rhs = np.concatenate(rhs_parts)
    rtype = np.concatenate(rtype_parts)
    A = coo_matrix((vals, (rows_i, rows_j)), shape=(r, n_vars)).tocsr()

    # ---- variable bounds & types ---------------------------------------- #
    lb = np.zeros(n_vars, dtype="float64")
    ub = np.ones(n_vars, dtype="float64")
    vtype = np.array(["I"] * n_vars)                       # all binary

    uidx = {uid: u for u, uid in enumerate(avail_unit_ids)}

    # max_reloc_min (HARD): a unit may not relocate farther than this drive-time
    # from its current post. Unreachable pairs (1e6 sentinel) are always forbidden.
    # Applied before lock/force so an explicit force_station can still override.
    max_reloc = constraints["max_reloc_min"]
    too_far = (reloc_uS > max_reloc) if max_reloc is not None else np.zeros((U, S), bool)
    forbid = too_far | (reloc_uS >= C.UNREACHABLE_MIN / 2)     # (U, S)
    forbid[np.arange(U), home_arr] = False                    # never forbid staying home
    ubx = ub[:US].reshape(U, S)
    ubx[forbid] = 0.0
    ub[:US] = ubx.reshape(-1)

    # lock_units: pin to current home station
    for uid in constraints["lock_units"]:
        if uid in uidx:
            u = uidx[uid]; home = home_station[u]
            for s in range(S):
                lb[build.xi(u, s)] = 1.0 if s == home else 0.0
                ub[build.xi(u, s)] = 1.0 if s == home else 0.0

    # force_station: pin to a chosen station
    for uid, s_force in constraints["force_station"].items():
        if uid in uidx:
            u = uidx[uid]
            for s in range(S):
                fix = 1.0 if s == s_force else 0.0
                lb[build.xi(u, s)] = fix; ub[build.xi(u, s)] = fix

    # protect_zones: force coverage  (y[z] = 1 -> link forces a covering unit)
    fsa_to_z = {world["fsa_index"][z]: z for z in range(Z)}
    for fsa in constraints["protect_zones"]:
        z = fsa_to_z.get(fsa)
        if z is not None:
            lb[build.yi(z)] = 1.0

    # ---- assemble cuOpt DataModel --------------------------------------- #
    dm = DataModel()
    dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
    dm.set_constraint_bounds(np.asarray(rhs, dtype="float64"))
    dm.set_row_types(np.asarray(rtype))
    dm.set_objective_coefficients(obj)
    dm.set_variable_types(vtype)
    dm.set_variable_lower_bounds(lb)
    dm.set_variable_upper_bounds(ub)
    dm.set_maximize(True)

    build.data_model = dm
    return build
