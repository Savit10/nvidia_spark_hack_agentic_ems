"""
Integration loop — Person 3 owns the center of the system.

Each decision tick:
  1. advance the sim to the tick (calls arrive, units go busy/free),
  2. snapshot State and `evaluate` current coverage,
  3. optionally turn an operator's English command into Constraints (the LLM),
  4. `re_optimize` to get relocation moves (cuOpt),
  5. `explain_result` in plain English (the LLM),
  6. apply the moves back into the sim.

re_optimize is imported lazily so the loop still runs (coverage-only) on a box
without cuOpt — it just reports that optimization was skipped.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import contracts as C
from optimizer import evaluate
from sim_agent import agent
from sim_agent.engine import Simulation


# Per-command latency breakdown log (bottleneck investigation). Appended to on
# every operator-command step so you can see which stage dominates — LLM parse,
# cuOpt solve, or LLM explain.  tail -f latency.log to watch it live.
_LATENCY_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latency.log")


def _log_timings(tr: "TickResult") -> None:
    try:
        t = tr.timings or {}
        order = ["sim_advance_ms", "parse_intent_ms", "dispatch_ms",
                 "evaluate_before_ms", "re_optimize_ms", "apply_moves_ms", "explain_ms"]
        stages = " ".join(f"{k[:-3]}={t.get(k, 0):.0f}" for k in order)
        line = (f"{datetime.now().isoformat(timespec='milliseconds')} | "
                f"cmd={tr.command!r} | intent={tr.intent} | "
                f"parse_src={agent.LAST_PARSE_SOURCE} explain_src={agent.LAST_EXPLAIN_SOURCE} | "
                f"{stages} | TOTAL={t.get('total_ms', 0):.0f} ms\n")
        with open(_LATENCY_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass  # logging must never break a request


@dataclass
class TickResult:
    t: float
    coverage_before: C.CoverageMap
    command: Optional[str]
    constraints: Optional[C.Constraints]
    result: Optional[C.OptimizeResult]
    applied: list[dict]
    explanation: str
    calls_this_tick: int
    calls: list[dict] = field(default_factory=list)   # enriched incident records this tick
    intent: str = "coverage"                          # "coverage" | "emergency"
    dispatched: Optional[dict] = None                 # emergency dispatch summary (units sent to scene)
    timings: dict = field(default_factory=dict)       # per-stage latency (ms) for this step


@dataclass
class Loop:
    sim: Simulation
    world: C.World
    tick_min: float = 15.0
    optimize: bool = True
    history: list[TickResult] = field(default_factory=list)

    def _re_optimize(self, state, constraints):
        """Lazy cuOpt call; returns (result, error_or_None)."""
        try:
            from optimizer import re_optimize
        except Exception as e:  # cuOpt not installed
            return None, f"optimizer import failed: {e}"
        try:
            return re_optimize(state, self.world, constraints), None
        except Exception as e:  # GPU/solver error at call time
            return None, f"re_optimize failed: {e}"

    def step(self, command: Optional[str] = None,
             base_constraints: Optional[C.Constraints] = None,
             explain_llm: Optional[bool] = None) -> TickResult:
        # By default narrate with the LLM only when the operator issued a command
        # (the high-value moment); auto-play ticks use the instant template.
        if explain_llm is None:
            explain_llm = command is not None
        timings: dict = {}
        _t_total = time.perf_counter()

        target = self.sim.t + self.tick_min
        _t = time.perf_counter()
        calls = self.sim.run_until(target)
        timings["sim_advance_ms"] = (time.perf_counter() - _t) * 1000.0

        # Classify the operator's intent: an active EMERGENCY (send units to a scene
        # now) vs a COVERAGE request (reposition idle units). For an emergency we
        # commit the fastest units to the scene FIRST, then re-optimize the depleted
        # fleet for coverage — so cuOpt is restructuring around the real dispatch.
        # NOTE: parse_intent makes the LLM round-trip(s) when a command is present.
        _t = time.perf_counter()
        intent = (agent.parse_intent(command, self.world) if command
                  else {"intent": "coverage", "dispatch": None, "constraints": {}})
        timings["parse_intent_ms"] = (time.perf_counter() - _t) * 1000.0

        dispatched = None
        _t = time.perf_counter()
        if intent.get("intent") == "emergency" and intent.get("dispatch"):
            d = intent["dispatch"]
            dispatched = self.sim.dispatch_incident(
                d["zone"], d.get("n_units", 2), d.get("priority", "DELTA"),
                d.get("reason", "major incident"))
            calls = calls + dispatched["records"]      # animate the dispatch trips/blips
        timings["dispatch_ms"] = (time.perf_counter() - _t) * 1000.0

        state = self.sim.snapshot()                    # post-dispatch: fleet now depleted

        # base_constraints come from the UI (e.g. a max-relocation slider) and act
        # as defaults; a coverage command's parsed constraints override them.
        parsed = intent.get("constraints") or {}
        merged = {**(base_constraints or {}), **(parsed or {})}
        constraints = merged or None

        # Evaluate "before" at the same threshold the optimizer will use, so the
        # before->after delta shown to the operator is apples-to-apples even when
        # a command overrides threshold_min.
        threshold = (constraints or {}).get("threshold_min") or self.world["threshold_min"]
        _t = time.perf_counter()
        before = evaluate(state, self.world, threshold)
        timings["evaluate_before_ms"] = (time.perf_counter() - _t) * 1000.0

        n_available = sum(1 for u in state["units"] if u["status"] == "available")

        result = None
        applied = []
        timings["re_optimize_ms"] = 0.0
        timings["apply_moves_ms"] = 0.0
        timings["explain_ms"] = 0.0
        if not self.optimize:
            explanation = (
                f"Coverage {before['covered_demand_pct']*100:.0f}% of demand; "
                f"{len(before['gaps'])} gap zones."
            )
        elif n_available == 0:
            explanation = (
                f"All {len(state['units'])} units are busy on calls — none "
                f"available to relocate. Coverage is "
                f"{before['covered_demand_pct']*100:.0f}% of demand until a unit frees."
            )
        else:
            _t = time.perf_counter()
            result, err = self._re_optimize(state, constraints)   # cuOpt solve
            timings["re_optimize_ms"] = (time.perf_counter() - _t) * 1000.0
            if result is None:
                explanation = f"(optimization skipped) {err}"
            else:
                _t = time.perf_counter()
                applied = self.sim.apply_moves(result["moves"])
                timings["apply_moves_ms"] = (time.perf_counter() - _t) * 1000.0
                _t = time.perf_counter()
                explanation = agent.explain_result(result, self.world, use_llm=explain_llm)  # LLM explain
                timings["explain_ms"] = (time.perf_counter() - _t) * 1000.0

        if dispatched is not None:
            units = ", ".join(dispatched["dispatched"]) or "no free units"
            resp = "/".join(f"{r:.0f}" for r in dispatched["response_min"])
            short = (f" ({dispatched['shortfall']} more requested than were free)"
                     if dispatched["shortfall"] else "")
            prefix = (f"🚨 EMERGENCY in {dispatched['zone']} ({dispatched['reason']}): "
                      f"dispatched {len(dispatched['dispatched'])} unit(s) {units}"
                      + (f" (response {resp} min)" if resp else "") + short
                      + ". cuOpt then restructured the remaining fleet for coverage — ")
            explanation = prefix + explanation

        timings["total_ms"] = (time.perf_counter() - _t_total) * 1000.0

        tr = TickResult(
            t=self.sim.t,
            coverage_before=before,
            command=command,
            constraints=constraints,
            result=result,
            applied=applied,
            explanation=explanation,
            calls_this_tick=len(calls),
            calls=calls,
            intent=intent.get("intent", "coverage"),
            dispatched=dispatched,
            timings=timings,
        )
        self.history.append(tr)
        if command:                       # log the operator-command latency breakdown
            _log_timings(tr)
        return tr

    def run(
        self,
        n_ticks: int,
        commands: Optional[dict[int, str]] = None,
        on_tick: Optional[Callable[[int, TickResult], None]] = None,
    ) -> list[TickResult]:
        """Run n_ticks decision steps. `commands` maps tick-index -> English."""
        commands = commands or {}
        out = []
        for k in range(n_ticks):
            tr = self.step(command=commands.get(k))
            out.append(tr)
            if on_tick:
                on_tick(k, tr)
        return out
