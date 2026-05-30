"""
Demo frontend (temporary — three.js version comes later).

    source ~/cuopt-env/bin/activate
    streamlit run frontend/app.py

Drives the REAL integrated system (all three lanes):
  Person 3 Simulation  ->  State
  Person 3 agent.parse_command  ->  Constraints   (Nemotron NIM, rule fallback)
  Person 2 re_optimize (cuOpt)  ->  moves
  Person 3 agent.explain_result ->  English        (Nemotron NIM, template fallback)

Type a dispatcher command, hit Step, watch the red gaps heal and read the
agent's explanation. No NIM required — the agent falls back to rules/templates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

from load_world import load_world                         # noqa: E402
from optimizer import evaluate                            # noqa: E402
from sim_agent import agent                               # noqa: E402
from sim_agent.engine import Simulation                   # noqa: E402
from sim_agent.loop import Loop                           # noqa: E402
from audio.asr_client import asr_available, asr_info, transcribe_bytes  # noqa: E402
from audio.dispatch_clips import DISPATCH_CLIPS           # noqa: E402
from audio.normalize import normalize_dispatch_text       # noqa: E402

WORLD_NPZ = ROOT / "datasets/world.npz"
WORLD_META = ROOT / "datasets/world_meta.json"
GEOJSON = ROOT / "artifacts/toronto_fsa.geojson"
CLIPS_DIR = ROOT / "audio/clips"


@st.cache_resource
def get_world():
    return load_world(str(WORLD_NPZ), str(WORLD_META))


@st.cache_resource
def get_geojson():
    return json.loads(GEOJSON.read_text())


# --------------------------------------------------------------------------- #
#  map rendering
# --------------------------------------------------------------------------- #
def color_geojson(geo, cov_map):
    pz = cov_map["per_zone"]
    dem = np.array([v["demand_weight"] for v in pz.values()]) if pz else np.array([1.0])
    dmax = float(dem.max()) or 1.0
    feats = []
    for f in geo["features"]:
        z = pz.get(f["properties"]["fsa"])
        if z is None:
            color = [120, 120, 120, 40]
        else:
            a = int(70 + 170 * (z["demand_weight"] / dmax))
            color = [40, 180, 90, a] if z["covered"] else [220, 50, 50, a]
        feats.append({**f, "properties": {**f["properties"], "color": color,
                      "covered": "yes" if (z and z["covered"]) else "no",
                      "nearest": f"{z['nearest_unit_min']:.1f}" if z else "n/a"}})
    return {"type": "FeatureCollection", "features": feats}


def layers(world, state, cov_map, moves):
    sm = world["station_meta"]
    upts = [{"lon": sm[u["station"]]["lon"], "lat": sm[u["station"]]["lat"],
             "id": u["unit_id"],
             "color": [60, 220, 120] if u["status"] == "available" else [255, 140, 0]}
            for u in state["units"]]
    spts = [{"lon": s["lon"], "lat": s["lat"], "name": s["name"]} for s in sm.values()]
    arcs = [{"from": [sm[m["from"]]["lon"], sm[m["from"]]["lat"]],
             "to": [sm[m["to"]]["lon"], sm[m["to"]]["lat"]]}
            for m in moves if m.get("applied", True)]
    return [
        pdk.Layer("GeoJsonLayer", color_geojson(get_geojson(), cov_map), stroked=True,
                  filled=True, get_fill_color="properties.color",
                  get_line_color=[255, 255, 255, 60], line_width_min_pixels=0.5, pickable=True),
        pdk.Layer("ScatterplotLayer", spts, get_position=["lon", "lat"],
                  get_fill_color=[30, 30, 30, 160], get_radius=110),
        pdk.Layer("ScatterplotLayer", upts, get_position=["lon", "lat"],
                  get_fill_color="color", get_radius=330, pickable=True),
        pdk.Layer("ArcLayer", arcs, get_source_position="from", get_target_position="to",
                  get_width=4, get_source_color=[255, 215, 0], get_target_color=[0, 150, 255]),
    ]


# --------------------------------------------------------------------------- #
#  app
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Ambulance Coverage Optimizer", layout="wide")
world = get_world()

st.sidebar.title("🚑 Simulation")
n_units = st.sidebar.slider("Units in service", 4, min(40, world["S"]), 12)
calls_ph = st.sidebar.slider("Calls per hour", 5, 60, 18)
service = st.sidebar.slider("Mean service (min)", 15, 90, 40)
tick_min = st.sidebar.slider("Minutes per Step", 5, 60, 15)
seed = st.sidebar.number_input("Seed", 0, 9999, 0, step=1)

st.sidebar.title("🧭 Relocation policy")
max_reloc = st.sidebar.slider("Max relocation (min)", 3.0, 30.0, 10.0, 0.5,
                              help="A unit won't be repositioned farther than this drive-time. "
                                   "Tighten it to keep moves local; loosen for more coverage.")
familiar = st.sidebar.slider("Familiar distance (min)", 1.0, 15.0, 5.0, 0.5,
                             help="Relocations up to this length are 'cheap' (familiar turf); "
                                  "longer hauls cost progressively more in the objective.")

nim = agent.nim_available()
st.sidebar.markdown(f"**Agent:** {'🟢 Nemotron NIM' if nim else '🟡 rule/template fallback'}")

# (re)build the sim+loop when params change
sig = (n_units, calls_ph, service, tick_min, int(seed))
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    sim = Simulation(world, n_units=n_units, calls_per_hour=calls_ph,
                     mean_service_min=service, seed=int(seed))
    st.session_state.loop = Loop(sim, world, tick_min=tick_min)
    st.session_state.last = None

loop: Loop = st.session_state.loop
last = st.session_state.get("last")

st.title("Ambulance Relocation — live coverage")
cmd = st.text_input(
    "Dispatcher command (plain English, optional)",
    placeholder="e.g. Increase coverage around M5V, it's quiet out in M1B — pull from there",
    help="Soft steering: 'boost/increase coverage in M5V', 'ease off M1B' (reweights "
         "priorities, units redirect from quiet zones to busy ones). "
         "Hard levers: 'guarantee M4T stays covered' (must-cover), "
         "'evacuate M1B' (vacate). Also: 'no more than 2 moves', 'keep AMB_03', "
         "'use a 7 minute threshold'.")
# relocation policy from the sidebar sliders; a typed command can still override
base_constraints = {"max_reloc_min": max_reloc, "familiar_min": familiar}


def _spinner_msg(has_cmd: bool) -> str:
    """What the spinner says while a Step runs. Calls out the LLM stage when an
    operator command is being interpreted (Nemotron NIM, or the rule fallback)."""
    if has_cmd:
        engine = "Nemotron NIM" if nim else "rule parser"
        return (f"🧠 Agent thinking — interpreting your command ({engine}), "
                f"optimizing relocations, and writing the explanation…")
    return "⚙️ Advancing the sim and optimizing relocations…"


def _done_label(tr) -> str:
    n = tr.result["n_moves"] if (tr and tr.result) else 0
    return f"✅ Done — {n} relocation(s) computed"


b1, b2, b3 = st.columns([1, 1, 1])
if b1.button("▶ Step (advance + optimize)", type="primary", use_container_width=True):
    has_cmd = bool(cmd and cmd.strip())
    with st.status(_spinner_msg(has_cmd), expanded=True) as status:
        st.session_state.last = loop.step(command=cmd or None, base_constraints=base_constraints)
        status.update(label=_done_label(st.session_state.last), state="complete", expanded=False)
    last = st.session_state.last
    st.toast(_done_label(last), icon="🚑")
if b2.button("⏩ Step ×5", use_container_width=True):
    with st.status("⚙️ Running 5 steps and optimizing relocations…", expanded=True) as status:
        for i in range(5):
            status.update(label=f"⚙️ Step {i+1}/5 — advancing sim and optimizing…")
            st.session_state.last = loop.step(base_constraints=base_constraints)
        status.update(label="✅ 5 steps done", state="complete", expanded=False)
    last = st.session_state.last
if b3.button("↺ Reset", use_container_width=True):
    st.session_state.sig = None
    st.rerun()

# --- 📻 dispatch audio: play a call -> local ASR -> agent -> cuOpt ---------- #
asr_up = asr_available()
with st.container(border=True):
    info = asr_info()
    badge = (f"🟢 local ASR: {info.get('model','?')} on {info.get('device','?')}"
             if asr_up else "🟡 ASR service offline — start audio/asr_service.py in ~/asr-env")
    st.markdown(f"**📻 Dispatch radio** &nbsp; {badge}")
    cols = st.columns(len(DISPATCH_CLIPS))
    for col, clip in zip(cols, DISPATCH_CLIPS):
        wav = CLIPS_DIR / f"{clip['id']}.wav"
        with col:
            if wav.exists():
                st.audio(str(wav))
            disabled = not (asr_up and wav.exists())
            if st.button(clip["label"], key=f"clip_{clip['id']}", disabled=disabled,
                         use_container_width=True):
                with st.status("🎧 Transcribing dispatch audio (local ASR)…",
                               expanded=True) as status:
                    raw = transcribe_bytes(wav.read_bytes())
                    norm = normalize_dispatch_text(raw, world)
                    status.update(label=f"🧠 Heard “{norm}” — optimizing relocations…")
                    st.session_state.last = loop.step(command=norm, base_constraints=base_constraints)
                    status.update(label=_done_label(st.session_state.last),
                                  state="complete", expanded=False)
                st.session_state.last_audio = {"label": clip["label"], "raw": raw, "norm": norm}
                last = st.session_state.last
    la = st.session_state.get("last_audio")
    if la:
        st.caption(f"🎙️ heard: \"{la['raw']}\"  →  parsed command: `{la['norm']}`")

# current (post-step) state drives the live map
state = loop.sim.snapshot()
cov = evaluate(state, world, world["threshold_min"])
moves = last.applied if last else []
avail = sum(u["status"] == "available" for u in state["units"])

# KPI row
delta = None
if last and last.result:
    delta = (last.result["coverage_after"]["covered_demand_pct"]
             - last.result["coverage_before"]["covered_demand_pct"]) * 100
k = st.columns(5)
k[0].metric("Demand covered", f"{cov['covered_demand_pct']*100:.1f}%",
            delta=f"{delta:+.1f} pts" if delta is not None else None)
k[1].metric("Gap zones", len(cov["gaps"]))
k[2].metric("Units available", f"{avail}/{len(state['units'])}")
k[3].metric("Sim clock", f"{loop.sim.t:.0f} min")
kp = loop.sim.kpis()
k[4].metric("On-time", f"{kp['on_time_pct']*100:.0f}%",
            help=f"{kp['served']} served / {kp['calls']} calls · mean resp {kp['mean_response_min']:.1f} min")

# agent explanation + parsed constraints
if last:
    st.info(f"🗣️ **Agent:** {last.explanation}")
    if last.command and last.constraints:
        st.caption(f"parsed command → constraints: `{json.dumps(last.constraints)}`")
    # operator typed a command but it mapped to no zone/lever -> tell them why
    _zone_keys = {"protect_zones", "forbid_zones", "zone_priority",
                  "lock_units", "force_station", "max_moves", "threshold_min"}
    if last.command and last.command.strip() and not (set(last.constraints or {}) & _zone_keys):
        st.warning("Couldn't map that command to a zone or rule. Name an FSA postal "
                   "code (e.g. **M5V**), e.g. “cover M5V” or “ease off M1B”.")
    # escalation notes from re_optimize (relaxed cap / unreachable zone explanations)
    for note in (last.result or {}).get("notes", []):
        st.warning(f"⚠️ {note}")

st.pydeck_chart(pdk.Deck(
    map_style="road",
    initial_view_state=pdk.ViewState(latitude=43.70, longitude=-79.38, zoom=9.4, pitch=35),
    layers=layers(world, state, cov, moves),
    tooltip={"html": "<b>{fsa}</b><br/>covered: {covered}<br/>nearest unit: {nearest} min"},
))
st.caption("green = covered, red = gap (opacity ∝ call demand) · "
           "green dots = available units, orange = busy · arcs = this step's relocations")

if moves:
    st.subheader("Relocations this step")
    st.table([{"unit": m["unit_id"], "from": world["station_index"][m["from"]],
               "to": world["station_index"][m["to"]], "eta_min": m["eta_min"],
               "reason": m.get("reason", ""), "applied": m.get("applied", True)}
              for m in moves])
