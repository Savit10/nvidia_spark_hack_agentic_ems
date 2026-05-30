"""
Monte Carlo expected-coverage on the GPU — Person 2's "Spark story".

Current coverage is a single snapshot. What an operator actually cares about is:
"if the next wave of calls lands where calls *tend* to land, how much of that
demand can we still reach in time?" We answer that by sampling many demand
realizations and scoring placements against each — fully vectorized on the GPU.

Two entry points:
  * expected_coverage(state, world)         -> robustness of ONE placement
  * rank_plans([state_a, state_b, ...], ..) -> rank MANY placements at once

`rank_plans` is the workhorse: it scores every candidate placement against ONE
shared scenario matrix with a single (N x Z) @ (Z x P) GEMM, turning the toy
"score one mask" op into a GPU-sized ranking engine. Everything stays float32
and on-device; only a handful of summary scalars are copied back to the host.

No new contract: consumes State(s) + World, returns small stats dicts.
"""
from __future__ import annotations

import numpy as np

import contracts as C
from optimizer.evaluate import covered_mask

try:
    import cupy as cp
    _XP = cp
    _ON_GPU = True
except Exception:                       # pragma: no cover - CPU fallback
    cp = None
    _XP = np
    _ON_GPU = False


# --------------------------------------------------------------------------- #
#  demand sampling
# --------------------------------------------------------------------------- #
def _demand_base(world: C.World) -> np.ndarray:
    return np.array(
        [world["fsa_meta"][z]["demand_weight"] for z in range(world["Z"])],
        dtype="float64",
    )


def sample_demand(world: C.World, n_scenarios: int, concentration: float,
                  seed: int) -> "np.ndarray":
    """(n_scenarios, Z) float32 demand matrix, rows ~ Dirichlet around the
    historical shares and summing to 1. Lives on the GPU when cuPy is present."""
    Z = world["Z"]
    alpha = np.maximum(concentration * _demand_base(world), 1e-6)
    if _ON_GPU:
        rng = cp.random.default_rng(seed)
        gamma = rng.gamma(shape=cp.asarray(alpha), size=(n_scenarios, Z))
        samples = gamma / gamma.sum(axis=1, keepdims=True)
        return samples.astype(cp.float32)
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha, size=n_scenarios).astype("float32")


def _summary(scores) -> dict:
    """mean / p5 / p95 of a (N,) score vector, computed on-device; 3 scalars out."""
    if _ON_GPU:
        q = cp.percentile(scores, cp.asarray([5.0, 95.0]))
        return {"expected_coverage_pct": float(scores.mean()),
                "p5": float(q[0]), "p95": float(q[1])}
    return {"expected_coverage_pct": float(scores.mean()),
            "p5": float(np.percentile(scores, 5)),
            "p95": float(np.percentile(scores, 95))}


# --------------------------------------------------------------------------- #
#  single placement
# --------------------------------------------------------------------------- #
def expected_coverage(
    state: C.State,
    world: C.World,
    n_scenarios: int = 200_000,
    concentration: float = 30.0,
    threshold_min: float | None = None,
    seed: int = 0,
) -> dict:
    """Mean / p5 / p95 demand-weighted coverage of the CURRENT placement across
    `n_scenarios` sampled demand realizations — the headline GPU robustness number."""
    samples = sample_demand(world, n_scenarios, concentration, seed)      # (N, Z)
    mask = _XP.asarray(covered_mask(state, world, threshold_min).astype("float32"))
    scores = samples @ mask                                               # (N,)
    return {**_summary(scores), "n_scenarios": int(n_scenarios), "on_gpu": _ON_GPU}


# --------------------------------------------------------------------------- #
#  many placements — the ranking engine
# --------------------------------------------------------------------------- #
def _masks_from_states(states: list[C.State], world: C.World,
                       threshold_min: float | None) -> np.ndarray:
    """(P, Z) float32 covered-mask matrix, one row per candidate placement."""
    return np.stack(
        [covered_mask(s, world, threshold_min).astype("float32") for s in states]
    )


def rank_plans(
    states: list[C.State],
    world: C.World,
    n_scenarios: int = 200_000,
    concentration: float = 30.0,
    threshold_min: float | None = None,
    seed: int = 0,
    labels: list[str] | None = None,
) -> dict:
    """Score every candidate placement against ONE shared scenario matrix and
    rank them by expected demand coverage (ties/risk broken by p5).

    This is the GPU-filling workload: scoring is a single (N x Z) @ (Z x P) GEMM
    on float32 — one matmul ranks the whole portfolio.
    """
    if not states:
        return {"ranking": [], "best": None, "n_scenarios": int(n_scenarios),
                "n_plans": 0, "on_gpu": _ON_GPU}

    P = len(states)
    labels = labels or [f"plan_{i}" for i in range(P)]
    masks = _XP.asarray(_masks_from_states(states, world, threshold_min))   # (P, Z)
    samples = sample_demand(world, n_scenarios, concentration, seed)        # (N, Z)

    scores = samples @ masks.T                                             # (N, P)
    if _ON_GPU:
        mean = cp.asnumpy(scores.mean(axis=0))
        q = cp.asnumpy(cp.percentile(scores, cp.asarray([5.0, 95.0]), axis=0))
        p5, p95 = q[0], q[1]
    else:
        mean = scores.mean(axis=0)
        p5 = np.percentile(scores, 5, axis=0)
        p95 = np.percentile(scores, 95, axis=0)

    ranking = sorted(
        ({"label": labels[i], "index": i,
          "expected_coverage_pct": float(mean[i]),
          "p5": float(p5[i]), "p95": float(p95[i])} for i in range(P)),
        key=lambda d: (d["expected_coverage_pct"], d["p5"]),
        reverse=True,
    )
    return {
        "ranking": ranking,
        "best": ranking[0],
        "n_scenarios": int(n_scenarios),
        "n_plans": P,
        "on_gpu": _ON_GPU,
    }
