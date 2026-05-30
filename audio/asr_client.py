"""
Client for the local ASR service (CONTRACTS-style boundary).

Stdlib-only (urllib) so it adds no deps to cuopt-env. Mirrors how
sim_agent/agent.py talks to the Nemotron NIM: a small local inference service
on localhost, with a graceful "offline" signal so the UI can fall back to a
typed command.
"""
from __future__ import annotations

import json
import os
import urllib.request

ASR_URL = os.environ.get("ASR_URL", "http://localhost:8010")
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "60"))


def asr_available() -> bool:
    try:
        with urllib.request.urlopen(f"{ASR_URL}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def asr_info() -> dict:
    try:
        with urllib.request.urlopen(f"{ASR_URL}/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def transcribe_bytes(wav_bytes: bytes) -> str:
    """POST raw WAV bytes to the service, return the transcript text."""
    req = urllib.request.Request(
        f"{ASR_URL}/transcribe", data=wav_bytes,
        headers={"Content-Type": "audio/wav"}, method="POST")
    with urllib.request.urlopen(req, timeout=ASR_TIMEOUT) as r:
        return json.loads(r.read()).get("text", "").strip()


def transcribe_file(path: str) -> str:
    with open(path, "rb") as f:
        return transcribe_bytes(f.read())
