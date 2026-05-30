"""
Operator agent — Person 3.

Two directions, both crossing the LLM boundary as strict JSON (CONTRACTS.md §6):

  6a. English command  -> Constraints   (parse_command)
  6b. OptimizeResult   -> English        (explain_result)

Talks to a local Nemotron NIM over the OpenAI-compatible REST API using only the
stdlib (urllib) so no extra deps are needed. If the NIM is unreachable or returns
anything unparseable, both paths fall back to a deterministic rule-based engine,
so the demo always runs — the NIM just makes it smarter.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import contracts as C

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "http://localhost:8000/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
NIM_TIMEOUT = float(os.environ.get("NIM_TIMEOUT", "30"))

_FSA_RE = re.compile(r"\b([A-Z]\d[A-Z])\b")
_UNIT_RE = re.compile(r"\bAMB[_ ]?(\d{1,2})\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
#  Glossary the LLM needs to resolve names -> indices/codes (CONTRACTS.md §6a)
# --------------------------------------------------------------------------- #
def build_glossary(world: C.World) -> dict:
    """{station_name: s-index}, list of FSA codes, list of unit-id patterns."""
    return {
        "station_name_to_index": {
            world["station_meta"][s]["name"]: s for s in range(world["S"])
        },
        "station_id_to_index": {
            world["station_meta"][s]["station_id"]: s for s in range(world["S"])
        },
        "fsa_codes": list(world["fsa_index"]),
    }


# --------------------------------------------------------------------------- #
#  NIM transport (stdlib only)
# --------------------------------------------------------------------------- #
def nim_available() -> bool:
    try:
        req = urllib.request.Request(f"{NIM_BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _chat(messages: list[dict], temperature: float = 0.0) -> str:
    """One OpenAI-compatible chat completion. Raises on any transport error."""
    body = json.dumps(
        {"model": NIM_MODEL, "messages": messages, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        f"{NIM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=NIM_TIMEOUT) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
#  6a. English command -> Constraints
# --------------------------------------------------------------------------- #
_CONSTRAINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "lock_units": {"type": "array", "items": {"type": "string"}},
        "force_station": {"type": "object", "additionalProperties": {"type": "integer"}},
        "max_moves": {"type": ["integer", "null"]},
        "protect_zones": {"type": "array", "items": {"type": "string"}},
        "threshold_min": {"type": ["number", "null"]},
        "move_penalty": {"type": ["number", "null"]},
    },
}


def parse_command(text: str, world: C.World) -> C.Constraints:
    """Natural-language operator command -> Constraints object (§3)."""
    if not text or not text.strip():
        return {}
    if nim_available():
        try:
            return _parse_command_nim(text, world)
        except Exception:
            pass  # fall through to rules
    return _parse_command_rules(text, world)


def _parse_command_nim(text: str, world: C.World) -> C.Constraints:
    glossary = build_glossary(world)
    system = (
        "You translate an ambulance dispatcher's English command into a JSON "
        "Constraints object for a relocation optimizer. Respond with ONLY the "
        "JSON object, no prose. Schema:\n"
        + json.dumps(_CONSTRAINTS_SCHEMA)
        + "\nOmit fields the command does not mention. "
        "Resolve station names and FSA postal codes using this glossary:\n"
        + json.dumps(glossary)
    )
    raw = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": text}]
    )
    obj = _extract_json(raw)
    return _coerce_constraints(obj, world)


def _parse_command_rules(text: str, world: C.World) -> C.Constraints:
    """Deterministic fallback parser — regex over common dispatcher phrasings."""
    c: C.Constraints = {}
    low = text.lower()
    valid_fsa = set(world["fsa_index"])

    units = [f"AMB_{int(m):02d}" for m in _UNIT_RE.findall(text)]

    if any(w in low for w in ("keep", "lock", "don't move", "do not move", "leave")):
        if units:
            c["lock_units"] = units

    m = re.search(
        r"(?:no more than|more than|at most|max(?:imum)?|up to|only)\s+(\d+)\s*moves?",
        low,
    )
    if m:
        c["max_moves"] = int(m.group(1))

    if any(w in low for w in ("cover", "protect", "ensure", "make sure")):
        zones = [z for z in _FSA_RE.findall(text) if z in valid_fsa]
        if zones:
            c["protect_zones"] = zones

    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*min", low) or re.search(
        r"(\d+(?:\.\d+)?)\s*[- ]?min(?:ute)?\s+threshold", low
    )
    if m:
        c["threshold_min"] = float(m.group(1))

    return c


def _as_list(v) -> list:
    """Tolerate LLMs that emit a bare scalar where the schema wants a list
    (e.g. Nemotron returns "lock_units": "AMB_03" instead of ["AMB_03"])."""
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _as_int(v):
    """Accept int, float, or numeric string for integer fields; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    return None


def _as_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _coerce_constraints(obj: dict, world: C.World) -> C.Constraints:
    """Normalize the LLM's JSON into a clean Constraints object.

    Robust to the two things LLMs do wrong here: emitting a scalar where a list
    is expected, and emitting numbers as strings. Unknown/unusable fields are
    dropped rather than passed to the solver.
    """
    out: C.Constraints = {}
    valid_fsa = set(world["fsa_index"])

    lock = [str(u) for u in _as_list(obj.get("lock_units")) if u]
    if lock:
        out["lock_units"] = lock

    if isinstance(obj.get("force_station"), dict):
        fs = {}
        for k, v in obj["force_station"].items():
            iv = _as_int(v)
            if iv is not None and 0 <= iv < world["S"]:
                fs[str(k)] = iv
        if fs:
            out["force_station"] = fs

    mm = _as_int(obj.get("max_moves"))
    if mm is not None:
        out["max_moves"] = mm

    zones = [str(z) for z in _as_list(obj.get("protect_zones")) if str(z) in valid_fsa]
    if zones:
        out["protect_zones"] = zones

    thr = _as_float(obj.get("threshold_min"))
    if thr is not None:
        out["threshold_min"] = thr

    mp = _as_float(obj.get("move_penalty"))
    if mp is not None:
        out["move_penalty"] = mp

    return out


# --------------------------------------------------------------------------- #
#  6b. OptimizeResult -> English explanation
# --------------------------------------------------------------------------- #
def explain_result(result: C.OptimizeResult, world: C.World) -> str:
    """2-4 sentence operator-facing explanation of a relocation result."""
    payload = _explain_payload(result, world)
    if nim_available():
        try:
            return _explain_nim(payload)
        except Exception:
            pass
    return _explain_template(payload)


def _explain_payload(result: C.OptimizeResult, world: C.World) -> dict:
    name = lambda s: world["station_meta"][s]["name"]
    before = result["coverage_before"]["covered_demand_pct"]
    after = result["coverage_after"]["covered_demand_pct"]
    healed = sorted(
        set(result["coverage_before"]["gaps"]) - set(result["coverage_after"]["gaps"])
    )
    moves = [
        {
            "unit": m["unit_id"],
            "from": name(m["from"]),
            "to": name(m["to"]),
            "eta_min": m["eta_min"],
            "reason": m["reason"],
        }
        for m in result["moves"]
    ]
    return {
        "status": result["status"],
        "n_moves": result["n_moves"],
        "moves": moves,
        "before_pct": before,
        "after_pct": after,
        "gaps_healed": healed,
        "solve_time_ms": result["solve_time_ms"],
    }


def _explain_nim(payload: dict) -> str:
    system = (
        "You are an ambulance dispatch assistant. Given a JSON relocation result, "
        "write a concise 2-4 sentence explanation for the operator: what to move, "
        "and how coverage improves. Plain text, no JSON, no markdown."
    )
    return _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.3,
    ).strip()


def _explain_template(p: dict) -> str:
    if p["status"] == "Infeasible":
        return (
            "No feasible relocation under the given constraints — the protected "
            "zones can't all be covered by the available units. Relax the "
            "constraints or add a unit."
        )
    if p["n_moves"] == 0:
        return (
            f"No moves needed: coverage is already at {p['before_pct']*100:.0f}% "
            f"of demand. (solved in {p['solve_time_ms']:.0f} ms)"
        )
    parts = [
        f"{m['unit']}: {m['from']} → {m['to']} (~{m['eta_min']:.0f} min, {m['reason']})"
        for m in p["moves"]
    ]
    healed = (
        f" Heals gap{'s' if len(p['gaps_healed'])>1 else ''} "
        + ", ".join(p["gaps_healed"])
        + "."
        if p["gaps_healed"]
        else ""
    )
    return (
        f"Recommend {p['n_moves']} move(s): "
        + "; ".join(parts)
        + f". Demand coverage rises {p['before_pct']*100:.0f}% → "
        f"{p['after_pct']*100:.0f}%.{healed} "
        f"(solved in {p['solve_time_ms']:.0f} ms)"
    )


# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM response (handles code fences)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.lower().startswith("json") else raw
    start, depth = raw.find("{"), 0
    if start < 0:
        return {}
    for i in range(start, len(raw)):
        depth += (raw[i] == "{") - (raw[i] == "}")
        if depth == 0:
            return json.loads(raw[start : i + 1])
    return {}
