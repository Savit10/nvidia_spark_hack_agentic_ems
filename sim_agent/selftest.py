"""
Fast, no-GPU self-checks for the sim + agent lane (Person 3).

    python -m sim_agent.selftest

Exercises the discrete-event engine, the rule-based command parser, and the
explanation template against the frozen mocks. Does NOT touch cuOpt, so it runs
anywhere and is safe in CI.
"""
from __future__ import annotations

import contracts as C
from sim_agent.agent import (
    _coerce_constraints,
    _explain_template,
    _explain_payload,
    _parse_command_rules,
)
from sim_agent.engine import Simulation


def _check(name, cond):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_engine():
    print("engine:")
    world = C.mock_world()
    sim = Simulation(world, n_units=3, calls_per_hour=120.0, seed=1)
    sim.run_until(60.0)
    state = sim.snapshot()
    _check("state has contract keys", set(state) == {"t", "units"})
    _check("clock advanced", state["t"] >= 60.0)
    _check("some calls arrived", sim.stats["calls"] > 0)
    _check(
        "unit shape matches contract",
        all(set(u) == {"unit_id", "status", "station", "busy_until"} for u in state["units"]),
    )
    # apply a move to an available unit
    avail = [u for u in state["units"] if u["status"] == "available"]
    if avail:
        uid, dest = avail[0]["unit_id"], (avail[0]["station"] + 1) % world["S"]
        applied = sim.apply_moves([{"unit_id": uid, "from": avail[0]["station"], "to": dest}])
        _check("move applied to available unit", applied[0]["applied"])


def test_parser():
    print("parser (rule fallback):")
    world = C.mock_world()
    c = _parse_command_rules(
        "Keep AMB_00 where it is and make sure M3A stays covered, "
        "but don't make more than 2 moves.",
        world,
    )
    _check("locked AMB_00", c.get("lock_units") == ["AMB_00"])
    _check("max_moves=2", c.get("max_moves") == 2)
    _check("protect M3A", c.get("protect_zones") == ["M3A"])

    c2 = _parse_command_rules("Use a 7 minute threshold for this run.", world)
    _check("threshold override", c2.get("threshold_min") == 7.0)

    c3 = _parse_command_rules("nonsense with no real instruction", world)
    _check("empty command -> empty constraints", c3 == {})


def test_coerce():
    print("LLM output coercion (real-model sloppiness):")
    world = C.mock_world()
    fsa = world["fsa_index"][0]
    # Nemotron emitted scalars instead of lists, and a stringy number.
    obj = {"lock_units": "AMB_03", "protect_zones": fsa, "max_moves": "2"}
    c = _coerce_constraints(obj, world)
    _check("scalar lock_units -> list", c.get("lock_units") == ["AMB_03"])
    _check("scalar protect_zones -> list", c.get("protect_zones") == [fsa])
    _check("string max_moves -> int", c.get("max_moves") == 2)

    # garbage / out-of-range gets dropped, not passed to the solver
    bad = {"protect_zones": ["NOT_A_FSA"], "force_station": {"AMB_0": 9999}}
    cb = _coerce_constraints(bad, world)
    _check("invalid FSA dropped", "protect_zones" not in cb)
    _check("out-of-range station dropped", "force_station" not in cb)


def test_explain():
    print("explanation template:")
    world = C.mock_world()
    txt = _explain_template(_explain_payload(C.mock_result(), world))
    _check("mentions a move", "AMB_01" in txt)
    _check("mentions coverage gain", "→" in txt or "->" in txt or "%" in txt)

    infeasible = {**C.mock_result(), "status": "Infeasible", "moves": [], "n_moves": 0}
    txt2 = _explain_template(_explain_payload(infeasible, world))
    _check("infeasible explained", "feasible" in txt2.lower())


def main():
    print("=" * 60)
    print("sim_agent self-test (no GPU)")
    print("=" * 60)
    test_engine()
    test_parser()
    test_coerce()
    test_explain()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
