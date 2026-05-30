"""
Scripted dispatch radio calls for the demo.

Each clip is a short transcript written in dispatcher cadence that, once
transcribed by the local ASR and normalized, maps cleanly to a Constraints
object via the agent. FSA codes used here are verified present in the real
world.npz (M5V, M1B, M5J, M4W are all live Toronto FSAs).

`make_clips.py` renders these transcripts to WAV (TTS) once, into audio/clips/.
"""
from __future__ import annotations

# id, on-screen label, spoken transcript, and the constraints we EXPECT
# (expected is for the demo/verification only — the real parse runs live).
DISPATCH_CLIPS = [
    {
        "id": "collision_m5v",
        "label": "🚨 Collision — downtown core",
        "transcript": ("All units be advised, multi-vehicle collision in the "
                       "downtown core. Make sure M5V stays covered, and no more "
                       "than two moves."),
        "expected": {"protect_zones": ["M5V"], "max_moves": 2},
    },
    {
        "id": "transfer_amb3",
        "label": "🏥 Hold a unit for transfer",
        "transcript": ("Dispatch to all cars, keep ambulance three on station, "
                       "we have a pending hospital transfer."),
        "expected": {"lock_units": ["AMB_03"]},
    },
    {
        "id": "fire_m1b",
        "label": "🔥 Structure fire — east end",
        "transcript": ("Priority one, working structure fire in the east end. "
                       "Ensure M1B is covered and tighten response to within "
                       "seven minutes."),
        "expected": {"protect_zones": ["M1B"], "threshold_min": 7.0},
    },
    {
        "id": "rebalance_m5j",
        "label": "📻 Rebalance — financial district",
        "transcript": ("All units, reposition to hold the financial district. "
                       "Make sure M5J stays covered, no more than three moves."),
        "expected": {"protect_zones": ["M5J"], "max_moves": 3},
    },
]


def clip_by_id(cid: str) -> dict | None:
    return next((c for c in DISPATCH_CLIPS if c["id"] == cid), None)
