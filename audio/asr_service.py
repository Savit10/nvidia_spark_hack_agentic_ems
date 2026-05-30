"""
Local ASR inference service — NVIDIA Parakeet (NeMo) on the GB10 GPU.

Runs in the isolated asr-env (torch + nemo_toolkit[asr]), NOT in cuopt-env, so
NeMo's heavy deps never touch the working cuOpt stack. Loads the model once and
serves transcription over HTTP — the same local-inference pattern as the NIM,
which is exactly the DGX-Spark story: two local models (ASR + LLM) resident in
unified memory, no audio ever leaving the box.

Launch:
    ~/asr-env/bin/python audio/asr_service.py            # serves on :8010

Env:
    ASR_MODEL   default nvidia/parakeet-tdt-0.6b-v2
    ASR_PORT    default 8010
"""
from __future__ import annotations

import io
import os
import tempfile

import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, Request

MODEL_NAME = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
PORT = int(os.environ.get("ASR_PORT", "8010"))
TARGET_SR = 16000

app = FastAPI(title="dispatch-asr")
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _model
    if _model is None:
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        m = m.to(_device).eval()
        _model = m
    return _model


def _to_16k_mono_wav(raw: bytes) -> str:
    """Decode arbitrary WAV bytes -> 16k mono temp wav path (Parakeet wants 16k)."""
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)                      # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)                  # downmix to mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    # torchaudio.save needs TorchCodec in 2.12; soundfile writes WAV directly.
    sf.write(path, wav.squeeze(0).numpy(), TARGET_SR, subtype="PCM_16")
    return path


def _text_from_result(res) -> str:
    """NeMo transcribe returns list[str] or list[Hypothesis] across versions."""
    if not res:
        return ""
    first = res[0]
    return (first.text if hasattr(first, "text") else str(first)).strip()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": _device,
            "loaded": _model is not None}


@app.post("/transcribe")
async def transcribe(request: Request):
    raw = await request.body()
    path = _to_16k_mono_wav(raw)
    try:
        with torch.inference_mode():
            res = get_model().transcribe([path], batch_size=1)
        return {"text": _text_from_result(res)}
    finally:
        os.remove(path)


if __name__ == "__main__":
    print(f"[asr] loading {MODEL_NAME} on {_device} ...")
    get_model()
    print(f"[asr] ready on :{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
