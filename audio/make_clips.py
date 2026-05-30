"""
Render the scripted dispatch transcripts to WAV — one-time asset generation.

Uses Piper (local neural TTS) via its stable CLI. These WAVs are static demo
assets; the LIVE, GPU, NVIDIA model in the pipeline is Parakeet ASR. (We tried
NeMo TTS first but its pretrained FastPitch config is incompatible with NeMo
2.7's g2p path — not worth fighting for throwaway assets.)

Run in the asr-env, from the repo root:
    ~/asr-env/bin/python -m audio.make_clips
Output: audio/clips/<id>.wav
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from audio.dispatch_clips import DISPATCH_CLIPS

HERE = Path(__file__).resolve().parent
VOICE = HERE / "en_US-lessac-medium.onnx"
OUT = HERE / "clips"
PIPER = Path(sys.executable).parent / "piper"


def main():
    if not VOICE.exists():
        sys.exit(f"voice model missing: {VOICE}\n"
                 f"download with: {sys.executable} -m piper.download_voices en_US-lessac-medium")
    OUT.mkdir(parents=True, exist_ok=True)
    for clip in DISPATCH_CLIPS:
        out = OUT / f"{clip['id']}.wav"
        subprocess.run([str(PIPER), "-m", str(VOICE), "-f", str(out)],
                       input=clip["transcript"].encode(), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        kb = out.stat().st_size // 1024
        print(f"[tts] {out.name}  ({kb} KB)  \"{clip['transcript'][:55]}...\"")
    print(f"[tts] done — {len(DISPATCH_CLIPS)} clips in {OUT}")


if __name__ == "__main__":
    main()
