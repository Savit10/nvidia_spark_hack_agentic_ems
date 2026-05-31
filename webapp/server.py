"""
Live backend for the Gotham webapp — serves the static frontend AND a JSON API
that drives the REAL integrated system on each request:

    sim.step  ->  agent.parse_command (Nemotron)  ->  cuOpt re_optimize  ->
    agent.explain_result (Nemotron)  ->  applied moves

Stdlib only (http.server) so no extra deps. Single-threaded on purpose: every
request runs in the serve_forever thread, so cuOpt / CuPy keep one stable CUDA
context. Fine for a single-operator demo.

Run from repo root in the cuOpt env:
    source ~/cuopt-env/bin/activate
    python -m webapp.server                      # rule/template agent
    NIM_BASE_URL=http://localhost:11434/v1 NIM_MODEL=nemotron3:33b python -m webapp.server   # live Nemotron

Then forward port 8080 in VS Code (PORTS panel) and open it.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBDIR = os.path.join(ROOT, "webapp")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

# Default the agent at the local ollama Nemotron unless explicitly overridden, so
# `python -m webapp.server` ALWAYS uses the LLM and never silently falls back just
# because someone forgot to export NIM_BASE_URL. Must run before importing agent
# (agent reads these env vars at import time).
os.environ.setdefault("NIM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("NIM_MODEL", "nemotron3:33b")

from load_world import load_world  # noqa: E402
from sim_agent import agent  # noqa: E402
from sim_agent.engine import Simulation  # noqa: E402
from sim_agent.loop import Loop  # noqa: E402
from webapp.generate_timeline import build_world_json, tick_record  # noqa: E402

# Dispatch-audio pipeline (Parakeet ASR). asr_client is stdlib-only (urllib), so
# importing it here is safe in cuopt-env; it just talks to the ASR service on :8010.
from audio.asr_client import asr_available, transcribe_file  # noqa: E402
try:
    from audio.dispatch_clips import DISPATCH_CLIPS  # noqa: E402
except Exception:
    DISPATCH_CLIPS = []
CLIPS_DIR = os.path.join(ROOT, "audio", "clips")

# --------------------------------------------------------------------------- #
WORLD = load_world(os.path.join(ROOT, "datasets", "world.npz"),
                   os.path.join(ROOT, "datasets", "world_meta.json"))

DEFAULTS = {"units": 16, "calls_per_hour": 20.0, "service": 32.0,
            "tick_min": 10.0, "seed": 7}

_LOCK = threading.Lock()
_STATE: dict = {"loop": None, "params": dict(DEFAULTS), "tick": 0}


def _new_loop(p: dict) -> Loop:
    sim = Simulation(WORLD, n_units=int(p["units"]),
                     calls_per_hour=float(p["calls_per_hour"]),
                     mean_service_min=float(p["service"]), seed=int(p["seed"]))
    return Loop(sim, WORLD, tick_min=float(p["tick_min"]))


_STATE["loop"] = _new_loop(_STATE["params"])


def do_step(command: str | None) -> dict:
    with _LOCK:
        tr = _STATE["loop"].step(command=command or None)
        rec = tick_record(WORLD, _STATE["loop"], tr)
        _STATE["tick"] += 1
        rec["tick_no"] = _STATE["tick"]
    return rec


def list_clips() -> list[dict]:
    """Dispatch clips that have a rendered WAV on disk -> launcher entries."""
    out = []
    for c in DISPATCH_CLIPS:
        if os.path.exists(os.path.join(CLIPS_DIR, f"{c['id']}.wav")):
            out.append({"id": c["id"], "label": c.get("label", c["id"]),
                        "audio_url": f"/clips/{c['id']}.wav"})
    return out


def do_dispatch(clip_id: str) -> dict:
    """Transcribe a dispatch clip with the local Parakeet ASR service (:8010).

    Returns the transcript the operator 'said' on the radio; the frontend then
    feeds it to /api/step as a command (parse -> cuOpt -> explain)."""
    cid = os.path.basename(str(clip_id or ""))            # no path traversal
    wav = os.path.join(CLIPS_DIR, f"{cid}.wav")
    if not os.path.exists(wav):
        return {"error": f"clip not found: {cid}"}
    if not asr_available():
        return {"error": "ASR offline — start audio/asr_service.py on :8010"}
    return {"clip_id": cid, "transcript": transcribe_file(wav)}


def do_reset(body: dict) -> dict:
    with _LOCK:
        p = {**DEFAULTS, **{k: body[k] for k in DEFAULTS if k in body}}
        _STATE["params"] = p
        _STATE["loop"] = _new_loop(p)
        _STATE["tick"] = 0
    return {"ok": True, "params": p}


# --------------------------------------------------------------------------- #
class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # quiet; keep startup banner only
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/world":
            return self._send_json(build_world_json(WORLD))
        if path == "/api/health":
            return self._send_json({"ok": True, "live": True,
                                    "nim": agent.nim_available(),
                                    "asr": asr_available(),
                                    "params": _STATE["params"]})
        if path == "/api/clips":
            return self._send_json(list_clips())
        if path.startswith("/clips/") and path.endswith(".wav"):
            return self._serve_wav(os.path.basename(path))
        return super().do_GET()

    def _serve_wav(self, name):
        fp = os.path.join(CLIPS_DIR, os.path.basename(name))
        if not os.path.exists(fp):
            return self._send_json({"error": "clip not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(os.path.getsize(fp)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(fp, "rb") as f:
            self.wfile.write(f.read())

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/api/step":
                return self._send_json(do_step(body.get("command")))
            if path == "/api/dispatch":
                return self._send_json(do_dispatch(body.get("clip_id")))
            if path == "/api/reset":
                return self._send_json(do_reset(body))
        except Exception as e:  # surface solver/agent errors to the client
            return self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        self._send_json({"error": "not found"}, 404)


def main():
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "127.0.0.1")
    httpd = HTTPServer((host, port), partial(Handler, directory=WEBDIR))
    print(f"LIVE backend on http://{host}:{port}  "
          f"(agent: {'Nemotron' if agent.nim_available() else 'rule/template fallback'})")
    print(f"  serving {WEBDIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
