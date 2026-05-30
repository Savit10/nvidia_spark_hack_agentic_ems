"""Throwaway integration audit — verifies the sim<->optimizer<->agent contract
boundaries hold at runtime on the REAL world. Run in ~/cuopt-env."""
import os, sys
import numpy as np
import contracts as C
from optimizer import evaluate
from sim_agent import Simulation, Loop
from sim_agent.agent import _parse_command_rules

DS = os.path.join(os.path.dirname(__file__), os.pardir, "datasets")
sys.path.insert(0, DS)
from load_world import load_world, validate_world

world = load_world(os.path.join(DS, "world.npz"), os.path.join(DS, "world_meta.json"))
validate_world(world)
S, Z = world["S"], world["Z"]

fails = []
def ck(name, cond, extra=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not cond: fails.append(name)

print("=" * 70)
print("INTEGRATION AUDIT (real world)")
print("=" * 70)

# ---- A. State shape/range produced by the sim is contract-legal -----------
print("\nA. sim State conforms to contract")
sim = Simulation(world, n_units=10, calls_per_hour=18.0, seed=3)
sim.run_until(30.0)
st = sim.snapshot()
ck("station indices in [0,S)", all(0 <= u["station"] < S for u in st["units"]))
ck("status domain", all(u["status"] in ("available", "busy") for u in st["units"]))
ck("busy<=>busy_until set",
   all((u["busy_until"] is None) == (u["status"] == "available") for u in st["units"]))

# ---- B. THE BIG ONE: optimizer's predicted coverage_after == sim reality ---
print("\nB. coverage_after predicted == evaluate(sim) after apply_moves")
loop = Loop(sim, world, tick_min=15.0)
for k in range(5):
    tr = loop.step()
    if tr.result is None:
        continue
    after_real = evaluate(loop.sim.snapshot(), world)
    pred = tr.result["coverage_after"]["covered_demand_pct"]
    real = after_real["covered_demand_pct"]
    ck(f"tick{k}: predicted {pred*100:.1f}% == applied {real*100:.1f}%",
       abs(pred - real) < 1e-6)

# ---- C. constraints from agent actually bind in the solver ----------------
print("\nC. parsed constraints change solver behavior")
sim2 = Simulation(world, n_units=12, calls_per_hour=30.0, seed=7)
sim2.run_until(45.0)
loop2 = Loop(sim2, world, tick_min=15.0)

state2 = loop2.sim.snapshot()
from optimizer import re_optimize
free = re_optimize(state2, world, None)
ck("unconstrained solve runs", free["status"] in ("Optimal", "Feasible"),
   f"n_moves={free['n_moves']}")

# max_moves cap
if free["n_moves"] >= 2:
    capped = re_optimize(state2, world, {"max_moves": 1})
    ck("max_moves=1 caps moves", capped["n_moves"] <= 1, f"n_moves={capped['n_moves']}")
else:
    print("  [skip] free solve made <2 moves; can't test cap meaningfully")

# lock_units: pick a unit the free solve MOVED, lock it, confirm it stays
moved = [m["unit_id"] for m in free["moves"]]
if moved:
    locked_id = moved[0]
    locked = re_optimize(state2, world, {"lock_units": [locked_id]})
    still_moving = [m["unit_id"] for m in locked["moves"]]
    ck(f"lock_units keeps {locked_id} home", locked_id not in still_moving)
else:
    print("  [skip] free solve made no moves; can't test lock")

# ---- D. agent command -> constraints maps to real indices/codes -----------
print("\nD. agent resolves a real command against the real world")
real_fsa = world["fsa_index"][10]
cmd = f"Keep AMB_03 where it is and make sure {real_fsa} stays covered, no more than 2 moves."
cons = _parse_command_rules(cmd, world)
ck("lock parsed", cons.get("lock_units") == ["AMB_03"])
ck("protect zone is a real FSA", cons.get("protect_zones") == [real_fsa])
ck("max_moves parsed", cons.get("max_moves") == 2)

# ---- E. relocation persists across ticks (no silent reset) ----------------
print("\nE. a relocated unit stays put until next decision")
sim3 = Simulation(world, n_units=8, calls_per_hour=12.0, seed=1)
loop3 = Loop(sim3, world, tick_min=15.0)
tr = loop3.step()
if tr.result and tr.result["moves"]:
    m = tr.result["moves"][0]
    u = next(x for x in loop3.sim.units if x["unit_id"] == m["unit_id"])
    ck("unit now at its move destination", u["station"] == m["to"],
       f"{m['unit_id']} at {u['station']} (to={m['to']})")
else:
    print("  [skip] no move on first tick")

print("\n" + "=" * 70)
print("AUDIT FAILURES:", fails if fails else "NONE — integration holds")
print("=" * 70)
sys.exit(1 if fails else 0)
