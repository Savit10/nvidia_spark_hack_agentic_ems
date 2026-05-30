# `audio/` — dispatch voice lane (NVIDIA Parakeet ASR)

Turns spoken dispatch radio calls into optimizer constraints, entirely locally:

```
🎙️ dispatch .wav  →  Parakeet ASR (GPU)  →  normalize  →  agent.parse_command  →  cuOpt  →  map heals
```

The ASR runs as a **local inference service** in its own env (`~/asr-env`,
torch+NeMo on the GB10), separate from `cuopt-env`, so NeMo's heavy deps never
touch the working cuOpt stack. The frontend calls it over HTTP — the same
local-service pattern as the Nemotron NIM. Two local models on one box is the
DGX-Spark story: private (audio never leaves the machine) and low-latency.

## Files
- `asr_service.py` — FastAPI service wrapping NeMo Parakeet. Runs in `~/asr-env`.
- `asr_client.py`  — stdlib HTTP client used by the frontend (`cuopt-env`).
- `normalize.py`   — spoken-language → command tokens (so the rule parser works
  even with the NIM offline; the LLM handles raw transcripts when it's up).
- `dispatch_clips.py` — the scripted demo calls (transcripts + expected parse).
- `make_clips.py`  — renders the transcripts to WAV via NeMo TTS (one-time).
- `clips/`         — generated dispatch WAVs.

## Run

```bash
# 1. one-time: generate the dispatch audio clips (in asr-env)
~/asr-env/bin/python -m audio.make_clips

# 2. start the ASR service (in asr-env) — loads Parakeet once, serves :8010
~/asr-env/bin/python audio/asr_service.py

# 3. run the frontend (in cuopt-env) — the 📻 Dispatch radio panel goes live
~/cuopt-env/bin/streamlit run frontend/app.py
```

Env knobs: `ASR_MODEL` (default `nvidia/parakeet-tdt-0.6b-v2`), `ASR_PORT`
(8010), and client-side `ASR_URL` / `ASR_TIMEOUT`.

If the service is down the panel shows 🟡 and the clip buttons disable — the
typed command box and the rest of the demo keep working.
