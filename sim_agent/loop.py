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

from dataclasses import dataclass, field
from typing import Callable, Optional

import contracts as C
from optimizer import evaluate
from sim_agent import agent
from sim_agent.engine import Simulation


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
        target = self.sim.t + self.tick_min
        calls = self.sim.run_until(target)

        # Classify the operator's intent: an active EMERGENCY (send units to a scene
        # now) vs a COVERAGE request (reposition idle units). For an emergency we
        # commit the fastest units to the scene FIRST, then re-optimize the depleted
        # fleet for coverage — so cuOpt is restructuring around the real dispatch.
        intent = (agent.parse_intent(command, self.world) if command
                  else {"intent": "coverage", "dispatch": None, "constraints": {}})
        dispatched = None
        if intent.get("intent") == "emergency" and intent.get("dispatch"):
            d = intent["dispatch"]
            dispatched = self.sim.dispatch_incident(
                d["zone"], d.get("n_units", 2), d.get("priority", "DELTA"),
                d.get("reason", "major incident"))
            calls = calls + dispatched["records"]      # animate the dispatch trips/blips

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
        before = evaluate(state, self.world, threshold)

        n_available = sum(1 for u in state["units"] if u["status"] == "available")

        result = None
        applied = []
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
            result, err = self._re_optimize(state, constraints)
            if result is None:
                explanation = f"(optimization skipped) {err}"
            else:
                applied = self.sim.apply_moves(result["moves"])
                explanation = agent.explain_result(result, self.world, use_llm=explain_llm)

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
        )
        self.history.append(tr)
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
