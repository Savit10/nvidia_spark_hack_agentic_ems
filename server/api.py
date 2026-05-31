"""
Live gateway for the cinematic frontend (runs in ~/cuopt-env).

    ~/cuopt-env/bin/uvicorn server.api:app --host 0.0.0.0 --port 8090
    # then open http://localhost:8090  (or the Tailscale IP)

Serves the webapp/ as static files (same origin -> no CORS) and exposes the
/api/* endpoints app.js already calls:

    GET  /api/health         -> {live, asr, nim}   (app.js flips to LIVE mode if live)
    GET  /api/world          -> stations / demand / bounds (build_world_json)
    POST /api/step {command} -> one live tick: sim advance + cuOpt + agent (tick_record + tick_no)
    GET  /api/clips          -> the scripted dispatch radio clips
    POST /api/clip/{id}      -> transcribe a clip via local Parakeet -> normalized command
    GET  /api/clip/{id}.wav  -> the clip audio (for the browser <audio> element)

Everything reuses the existing lanes — this is glue, not new logic. If cuOpt or
ASR is down a request degrades gracefully instead of 500-ing, so the demo holds.
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

# Default the agent at the local ollama Nemotron unless explicitly overridden, so
# the gateway uses the LLM even if NIM_BASE_URL was never exported (otherwise it
# silently falls back to the rule engine). Must precede the agent import.
os.environ.setdefault("NIM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("NIM_MODEL", "nemotron3:33b")

from load_world import load_world                                  # noqa: E402
from sim_agent import agent                                        # noqa: E402
from sim_agent.engine import Simulation                            # noqa: E402
from sim_agent.loop import Loop                                    # noqa: E402
from webapp.generate_timeline import build_world_json, tick_record  # noqa: E402
from audio import asr_client                                       # noqa: E402
from audio.dispatch_clips import DISPATCH_CLIPS, clip_by_id        # noqa: E402
from audio.normalize import normalize_dispatch_text                # noqa: E402

WEBAPP = os.path.join(ROOT, "webapp")
CLIPS = os.path.join(ROOT, "audio", "clips")

app = FastAPI(title="relocation-gateway")

# --- server-held sim/loop (built once) ------------------------------------- #
_world = load_world(os.path.join(ROOT, "datasets", "world.npz"),
                    os.path.join(ROOT, "datasets", "world_meta.json"))
_profiles = os.path.join(ROOT, "datasets", "incident_profiles.json")
_profiles = _profiles if os.path.exists(_profiles) else None
_state = {"loop": None, "tick_no": 0}


def _fresh_loop():
    sim = Simulation(_world, n_units=17, calls_per_hour=8.0, mean_service_min=30.0,
                     seed=5, incident_profiles=_profiles)
    return Loop(sim, _world, tick_min=10.0)


@app.on_event("startup")
def _startup():
    _state["loop"] = _fresh_loop()
    # warm up cuOpt so the first live click isn't a cold compile
    try:
        _state["loop"].step()
        _state["tick_no"] = 0
        _state["loop"] = _fresh_loop()      # reset clock after warm-up
    except Exception as e:                  # cuOpt unavailable -> live still serves coverage
        print(f"[gateway] warm-up skipped: {e}")


# --- API ------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"live": True, "asr": asr_client.asr_available(), "nim": agent.nim_available()}


@app.get("/api/world")
def world():
    return build_world_json(_world)


@app.post("/api/step")
async def step(body: dict | None = None):
    body = body or {}
    command = (body.get("command") or "").strip() or None
    try:
        tr = _state["loop"].step(command=command)
        _state["tick_no"] += 1
        rec = tick_record(_world, _state["loop"], tr, scene="LIVE")
        rec["tick_no"] = _state["tick_no"]
        return rec
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=200)


@app.post("/api/reset")
def reset():
    _state["loop"] = _fresh_loop()
    _state["tick_no"] = 0
    return {"ok": True}


@app.get("/api/clips")
def clips():
    return [{"id": c["id"], "label": c["label"], "transcript": c["transcript"]}
            for c in DISPATCH_CLIPS]


@app.get("/api/clip/{clip_id}.wav")
def clip_wav(clip_id: str):
    path = os.path.join(CLIPS, f"{clip_id}.wav")
    if not os.path.exists(path):
        return JSONResponse({"error": "clip not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/clip/{clip_id}")
def transcribe_clip(clip_id: str):
    """Local Parakeet ASR on a dispatch clip -> normalized operator command.
    Falls back to the clip's baked transcript if the ASR service is offline."""
    clip = clip_by_id(clip_id)
    if clip is None:
        return JSONResponse({"error": "unknown clip"}, status_code=404)
    path = os.path.join(CLIPS, f"{clip_id}.wav")
    transcript = clip["transcript"]
    used_asr = False
    if asr_client.asr_available() and os.path.exists(path):
        try:
            transcript = asr_client.transcribe_file(path)
            used_asr = True
        except Exception:
            pass
    return {"transcript": transcript, "asr": used_asr,
            "command": normalize_dispatch_text(transcript, _world)}


# --- static webapp (mounted last so /api/* wins) --------------------------- #
app.mount("/", StaticFiles(directory=WEBAPP, html=True), name="webapp")
