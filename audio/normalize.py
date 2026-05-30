"""
Dispatch-speech -> clean command text.

ASR returns natural spoken language ("keep ambulance three on station, no more
than two moves"). The rule-based parser in sim_agent/agent.py expects tokens
like AMB_03 and digit counts. This thin layer bridges the two so a spoken clip
works even when the Nemotron NIM is offline (the fallback path). When the NIM
IS live it parses the raw transcript fine — normalization just helps the rules.

Lives on the audio side so sim_agent/agent.py stays untouched.
"""
from __future__ import annotations

import re

import contracts as C

_WORD_NUM = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}
_UNIT_WORDS = r"(?:ambulance|ambo|unit|medic|car|rig|truck)"
_NUM = r"(?:\d{1,2}|" + "|".join(_WORD_NUM) + r")"


def _to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else _WORD_NUM[tok.lower()]


def normalize_dispatch_text(text: str, world: C.World | None = None) -> str:
    """Rewrite spoken dispatch language into the token forms the parser expects."""
    s = " " + text.strip() + " "

    # "ambulance three" / "unit 7" / "medic twelve" -> "AMB_03"
    def _unit(m):
        return f" AMB_{_to_int(m.group(1)):02d} "
    s = re.sub(_UNIT_WORDS + r"\s+(" + _NUM + r")", _unit, s, flags=re.IGNORECASE)

    # standalone number-words -> digits ("no more than two moves" -> "... 2 moves",
    # "within seven minutes" -> "within 7 minutes"). Done after unit handling.
    def _numword(m):
        return f" {_WORD_NUM[m.group(0).lower()]} "
    s = re.sub(r"\b(" + "|".join(_WORD_NUM) + r")\b", _numword, s, flags=re.IGNORECASE)

    # uppercase FSA postal tokens the ASR lower-cased ("m5v" -> "M5V")
    s = re.sub(r"\b([a-zA-Z]\d[a-zA-Z])\b", lambda m: m.group(1).upper(), s)

    return re.sub(r"\s+", " ", s).strip()
