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
# Bearer token for hosted NIMs / build.nvidia.com (nvapi-...). A bare local NIM
# container needs no auth, so this stays unset and no header is sent.
NIM_API_KEY = os.environ.get("NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")


def _auth_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if NIM_API_KEY:
        h["Authorization"] = f"Bearer {NIM_API_KEY}"
    return h

_FSA_RE = re.compile(r"\b([A-Za-z]\d[A-Za-z])\b")  # case-insensitive; .upper() the match
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
        req = urllib.request.Request(f"{NIM_BASE_URL}/models", headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _chat(messages: list[dict], temperature: float = 0.0,
          max_tokens: int = 256) -> str:
    """One OpenAI-compatible chat completion. Raises on any transport error.

    max_tokens is capped deliberately: a reasoning-style local model (e.g.
    nemotron-3-super) will otherwise emit a long trace and blow past the demo's
    latency budget. Parsing needs a tiny JSON; explaining needs 2-4 sentences."""
    body = json.dumps(
        {"model": NIM_MODEL, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens,
         # Disable the reasoning trace on reasoning models (e.g. nemotron-3-super
         # via Ollama). Without this the chain-of-thought fills the token budget
         # and lands in a separate `reasoning` field, leaving `content` empty.
         # Unknown to plain OpenAI servers, which ignore extra fields.
         "think": False}
    ).encode()
    req = urllib.request.Request(
        f"{NIM_BASE_URL}/chat/completions",
        data=body,
        headers=_auth_headers(),
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
        "forbid_zones": {"type": "array", "items": {"type": "string"}},
        "zone_priority": {"type": "object", "additionalProperties": {"type": "number"}},
        "threshold_min": {"type": ["number", "null"]},
        "move_penalty": {"type": ["number", "null"]},
    },
}

# How the operator's intent maps onto the levers. Soft steering (zone_priority)
# is the default; the hard levers are reserved for explicitly absolute language.
_INTENT_GUIDE = (
    "\nMapping the dispatcher's intent to fields:\n"
    "- INCREASE / boost / prioritize coverage in a zone (SOFT, preferred): add the "
    "FSA to zone_priority with a multiplier > 1 (≈2-3; use a bigger number for "
    "stronger wording). The optimizer will pull units toward it.\n"
    "- REDUCE / ease off / lower priority / it's quiet in a zone (SOFT, preferred): "
    "add the FSA to zone_priority with a multiplier < 1 (≈0.2-0.5). Units there "
    "become cheap to relocate, so they get freed to cover busier areas.\n"
    "- HARD guarantee only when the language is absolute ('must stay covered', "
    "'guarantee', 'no matter what'): put the FSA in protect_zones.\n"
    "- HARD vacate only for absolute language ('evacuate', 'clear out', 'pull "
    "everyone out of'): put the FSA in forbid_zones.\n"
    "Default to the soft zone_priority lever; reach for protect_zones/forbid_zones "
    "only when the operator clearly demands an absolute guarantee.\n"
)


# Records which path the most recent parse/explain actually took, so callers can
# show the operator whether Nemotron or the deterministic fallback was used.
LAST_PARSE_SOURCE: str = "none"
LAST_EXPLAIN_SOURCE: str = "none"


def parse_command(text: str, world: C.World) -> C.Constraints:
    """Natural-language operator command -> Constraints object (§3)."""
    global LAST_PARSE_SOURCE
    if not text or not text.strip():
        LAST_PARSE_SOURCE = "none"
        return {}
    if nim_available():
        try:
            out = _parse_command_nim(text, world)
            LAST_PARSE_SOURCE = "nemotron"
            return out
        except Exception:
            pass  # fall through to rules
    LAST_PARSE_SOURCE = "fallback"
    return _parse_command_rules(text, world)


# --------------------------------------------------------------------------- #
#  Intent: distinguish an active EMERGENCY (dispatch units to a scene now) from
#  a COVERAGE request (reposition idle units to keep an area ready). This is the
#  classification Nemotron is uniquely good at — emergencies and coverage commands
#  read similarly but trigger completely different actions downstream.
# --------------------------------------------------------------------------- #
_EMERGENCY_KW = (
    "collision", "crash", "mvc", "multi-vehicle", "multi vehicle", "pile-up", "pileup",
    "fire", "explosion", "blast", "mci", "mass casualty", "mass-casualty", "shooting",
    "stabbing", "derailment", "structure fire", "major incident", "active incident",
    "send units", "send help", "roll ", "dispatch ", "scene", "victims", "casualties",
)
_NUNITS_RE = re.compile(
    r"(?:send|roll|dispatch|need|get|want)\s+(\d+)"
    r"|(\d+)\s*(?:units|trucks|ambulances|medics|cars|crews|rigs|paramedics)",
    re.IGNORECASE,
)


def parse_intent(text: str, world: C.World) -> dict:
    """Classify the operator's command and return a structured intent:

        {"intent": "emergency"|"coverage",
         "dispatch": {"zone": FSA, "n_units": int, "reason": str, "priority": str} | None,
         "constraints": Constraints}

    'emergency' => commit units to a scene now (loop dispatches, then re-optimizes
    the rest). 'coverage' => the existing relocation-constraints path. Uses Nemotron
    when available, else a deterministic keyword fallback so the demo always runs."""
    global LAST_PARSE_SOURCE
    if not text or not text.strip():
        LAST_PARSE_SOURCE = "none"
        return {"intent": "coverage", "dispatch": None, "constraints": {}}
    if nim_available():
        try:
            out = _parse_intent_nim(text, world)
            LAST_PARSE_SOURCE = "nemotron"
            return out
        except Exception:
            pass
    LAST_PARSE_SOURCE = "fallback"
    return _parse_intent_rules(text, world)


_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["emergency", "coverage"]},
        "dispatch": {
            "type": ["object", "null"],
            "properties": {
                "zone": {"type": "string"},
                "n_units": {"type": "integer"},
                "reason": {"type": "string"},
                "priority": {"type": "string"},
            },
        },
        "constraints": _CONSTRAINTS_SCHEMA,
    },
}


def _parse_intent_nim(text: str, world: C.World) -> dict:
    glossary = {"fsa_codes": list(world["fsa_index"])}
    system = (
        "detailed thinking off\n"
        "You are an ambulance dispatch supervisor. Classify the operator's command "
        "and return ONLY a JSON object, no prose. Schema:\n"
        + json.dumps(_INTENT_SCHEMA)
        + "\nDecide intent:\n"
        "- 'emergency': an ACTIVE incident needing units sent to a scene NOW "
        "(collision, fire, mass-casualty, 'send/roll N units to <zone>'). Fill "
        "`dispatch` with the FSA zone, n_units (default 2 if unspecified, more for "
        "major incidents), a short reason, and a priority (DELTA/ECHO = "
        "life-threatening, CHARLIE/BRAVO/ALPHA = lower). Leave constraints empty.\n"
        "- 'coverage': a readiness/positioning request (keep/ensure/boost/ease an "
        "area's coverage). Set dispatch=null and fill `constraints`:\n"
        + _INTENT_GUIDE
        + "Resolve FSA postal codes using this glossary:\n"
        + json.dumps(glossary)
    )
    raw = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        max_tokens=400,
    )
    obj = _extract_json(raw)
    return _coerce_intent(obj, text, world)


def _coerce_intent(obj: dict, text: str, world: C.World) -> dict:
    valid_fsa = set(world["fsa_index"])
    d = obj.get("dispatch") if isinstance(obj.get("dispatch"), dict) else None
    if obj.get("intent") == "emergency" and d:
        zone = str(d.get("zone", "")).upper()
        if zone not in valid_fsa:
            zone = next((z.group(1).upper() for z in [_FSA_RE.search(text)] if z
                         and z.group(1).upper() in valid_fsa), None)
        n = _as_int(d.get("n_units")) or 2
        if zone:
            return {"intent": "emergency", "constraints": {},
                    "dispatch": {"zone": zone, "n_units": max(1, min(int(n), world["S"])),
                                 "reason": str(d.get("reason") or "major incident"),
                                 "priority": str(d.get("priority") or "DELTA").upper()}}
    return {"intent": "coverage", "dispatch": None,
            "constraints": _coerce_constraints(obj.get("constraints", obj), world)}


def _parse_intent_rules(text: str, world: C.World) -> dict:
    """Keyword fallback: emergency if incident words / 'send N units' appear."""
    low = text.lower()
    valid_fsa = set(world["fsa_index"])
    is_emergency = any(k in low for k in _EMERGENCY_KW)
    if is_emergency:
        m = _FSA_RE.search(text)
        zone = m.group(1).upper() if m and m.group(1).upper() in valid_fsa else None
        if zone:
            nm = _NUNITS_RE.search(text)
            n = int(next(g for g in nm.groups() if g)) if nm else (
                3 if any(w in low for w in ("multiple", "several", "mass", "mci", "major")) else 2)
            reason = next((k for k in _EMERGENCY_KW if k.strip() in low and len(k.strip()) > 4),
                          "major incident")
            return {"intent": "emergency", "constraints": {},
                    "dispatch": {"zone": zone, "n_units": max(1, min(n, world["S"])),
                                 "reason": reason, "priority": "DELTA"}}
    return {"intent": "coverage", "dispatch": None,
            "constraints": _parse_command_rules(text, world)}


def _parse_command_nim(text: str, world: C.World) -> C.Constraints:
    # Send ONLY the FSA codes, not the full station glossary. The big glossary
    # (every station name + id) blows up the prompt and makes smaller reasoning
    # models (e.g. nemotron3:33b) return empty content; FSA codes are all the
    # zone-based dispatcher commands need. Unit ids (AMB_NN) come straight from
    # the command text, so no glossary is required for lock_units.
    glossary = {"fsa_codes": list(world["fsa_index"])}
    system = (
        # Nemotron reasoning toggle: answer directly, no chain-of-thought trace
        # (keeps latency in the demo budget and avoids truncated/empty output).
        "detailed thinking off\n"
        "You translate an ambulance dispatcher's English command into a JSON "
        "Constraints object for a relocation optimizer. Respond with ONLY the "
        "JSON object, no prose. Schema:\n"
        + json.dumps(_CONSTRAINTS_SCHEMA)
        + _INTENT_GUIDE
        + "Omit fields the command does not mention. "
        "Resolve FSA postal codes using this glossary:\n"
        + json.dumps(glossary)
    )
    # max_tokens deliberately generous: even with think disabled, the model needs
    # headroom before it emits the JSON; too tight a cap yields empty content.
    raw = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        max_tokens=400,
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

    # Directional zone steering, parsed PER CLAUSE so a mixed command like
    # "boost M4T and ease off M1B" keeps the two directions separate (instead of
    # collapsing to one). Coverage verbs ("cover"/"make sure"/"guarantee") map to the
    # HARD protect lever, so the optimizer redirects units as needed to actually cover
    # the zone — re_optimize escalates the relocation cap to honor it, or explains why
    # it can't. Milder "boost"/"prioritize" stay on the soft zone_priority lever.
    HARD_VACATE = ("evacuate", "clear out", "pull everyone", "pull all units",
                   "vacate", "abandon")
    EASE = ("ease off", "reduce", "less coverage", "less", "lower", "deprioritize",
            "quiet", "pull back", "thin out", "slow", "calm")
    HARD_COVER = ("guarantee", "make sure", "ensure", "must", "no matter what",
                  "protect", "cover", "covered", "coverage", "keep", "critical",
                  "slammed", "swamped")
    BOOST = ("boost", "increase", "prioritize", "priority", "more units",
             "more ambulance", "send more", "focus", "reinforce", "lean")

    def _clause_dir(cl: str):
        if any(w in cl for w in HARD_VACATE):
            return "vacate"
        if any(w in cl for w in EASE):
            return "ease"
        if any(w in cl for w in HARD_COVER):
            return "cover"
        if any(w in cl for w in BOOST):
            return "boost"
        return None

    last_dir = None
    for clause in re.split(r"\b(?:and|but|then|also|while|;|,)\b", text):
        d = _clause_dir(clause.lower())
        if d:
            last_dir = d
        czones = [z.upper() for z in _FSA_RE.findall(clause) if z.upper() in valid_fsa]
        if not czones:
            continue
        use = d or last_dir or "cover"          # a bare "...M5V" defaults to coverage intent
        if use == "vacate":
            c.setdefault("forbid_zones", []).extend(czones)
        elif use == "cover":
            c.setdefault("protect_zones", []).extend(czones)
        elif use == "ease":
            c.setdefault("zone_priority", {}).update({z: 0.3 for z in czones})
        else:  # boost
            c.setdefault("zone_priority", {}).update({z: 3.0 for z in czones})

    for k in ("forbid_zones", "protect_zones"):     # de-dupe, preserve order
        if k in c:
            c[k] = list(dict.fromkeys(c[k]))

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

    zones = [str(z).upper() for z in _as_list(obj.get("protect_zones")) if str(z).upper() in valid_fsa]
    if zones:
        out["protect_zones"] = zones

    forbid = [str(z).upper() for z in _as_list(obj.get("forbid_zones")) if str(z).upper() in valid_fsa]
    if forbid:
        out["forbid_zones"] = forbid

    # zone_priority: {FSA: multiplier}. Keep only valid FSAs with a positive,
    # non-default multiplier; clamp to a sane band so an LLM can't emit a
    # multiplier that swamps or zeroes the whole objective.
    if isinstance(obj.get("zone_priority"), dict):
        zp = {}
        for k, v in obj["zone_priority"].items():
            fv = _as_float(v)
            if str(k).upper() in valid_fsa and fv is not None and fv > 0 and fv != 1.0:
                zp[str(k).upper()] = min(max(fv, 0.05), 10.0)
        if zp:
            out["zone_priority"] = zp

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
def explain_result(result: C.OptimizeResult, world: C.World, use_llm: bool = True) -> str:
    """2-4 sentence operator-facing explanation of a relocation result.

    `use_llm=False` forces the instant template (skips the Nemotron round-trip) —
    used for fast auto-play ticks; the LLM is reserved for operator commands."""
    global LAST_EXPLAIN_SOURCE
    payload = _explain_payload(result, world)
    if use_llm and nim_available():
        try:
            out = _explain_nim(payload)
            LAST_EXPLAIN_SOURCE = "nemotron"
            return out
        except Exception:
            pass
    LAST_EXPLAIN_SOURCE = "template"
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
        "gaps_after": list(result["coverage_after"]["gaps"]),
        "solve_time_ms": result["solve_time_ms"],
        "reasoning": result.get("reasoning"),
        "notes": result.get("notes", []),
        "decision": result.get("decision"),
    }


def _decision_story(p: dict) -> str:
    """Why cuOpt chose THESE units to satisfy the operator's command."""
    dec = p.get("decision")
    if not dec:
        return ""
    bits = []
    for d in dec:
        where = (f"{d['to']} — the only post within range of {d['zone']}"
                 if d["n_covering_stations"] == 1
                 else f"{d['to']} (1 of {d['n_covering_stations']} posts that cover {d['zone']})")
        why = (f"of {d['n_candidates']} available units it was the cheapest to relocate "
               f"({d['reloc_min']:.0f} min) to {where}")
        if d.get("cheapest") is False and d.get("counterfactual"):
            cf = d["counterfactual"]
            why = (f"cuOpt skipped the closer {cf['alt_unit']} ({cf['alt_reloc_min']:.0f} min) "
                   f"because using it covers only {cf['alt_total_pct']:.0f}% of demand vs "
                   f"{cf['chosen_total_pct']:.0f}% this way — {d['unit']} reaches {where}")
        elif len(d.get("candidates", [])) > 1:
            ru = d["candidates"][1]
            why += f"; next-closest {ru['unit']} was {ru['reloc_min']:.0f} min"
        bits.append(f"Chose {d['unit']} for {d['zone']}: {why}.")
    return " ".join(bits)


def _coverage_story(p: dict) -> str:
    """Plain-English account of how each operator-commanded zone got covered.

    Resolves the common confusion 'why didn't a unit move INTO the zone?': a zone
    is covered when an available unit is within the response-time threshold of it,
    which is usually a nearby post rather than the zone itself.
    """
    r = p.get("reasoning")
    if not r or not r.get("zones"):
        return ""
    thr = r["threshold_min"]
    bits = []
    for z in r["zones"]:
        fsa, by = z["fsa"], z.get("covered_by")
        if z["covered_after"] and by:
            verb = "now covered" if not z["covered_before"] else "stays covered"
            how = (f"{by['unit']} {'moved to' if by['moved'] else 'is posted at'} "
                   f"{by['station']}, {by['dist_min']:.0f} min from {fsa}")
            delta = (f" (response {z['before_min']:.0f}→{z['after_min']:.0f} min)"
                     if z["covered_before"] is False else "")
            bits.append(f"{fsa} {verb}: {how}{delta}")
        else:
            bits.append(f"{fsa} still uncovered — nearest unit {z['after_min']:.0f} min "
                        f"away (beyond the {thr:.0f}-min standard)")
    note_txt = (" " + " ".join(p.get("notes", []))) if p.get("notes") else ""
    return (" ".join(b + "." for b in bits)
            + f" (‘Covered’ means an available unit within {thr:.0f} min, "
            f"not a unit parked inside the zone.)" + note_txt)


def _explain_facts(p: dict) -> str:
    """Compact, pre-digested fact line for the LLM to phrase.

    We deliberately do NOT hand the model the full structured result (moves +
    reasoning + decision + counterfactual JSON): a reasoning model like
    nemotron-3-super will chew on that with a long chain-of-thought, blowing the
    latency budget and often emptying `content`. A short fact string keeps it to a
    fast, clean 2-sentence summary. The rich per-zone/decision detail still lives
    in _explain_template (used for auto-play ticks)."""
    moves = "; ".join(f"{m['unit']}->{m['to']} ({m['eta_min']:.0f}min)"
                      for m in p["moves"]) or "none"
    # Use the FULL post-move gap list (not just operator-commanded zones) so the
    # model can never claim "no zones uncovered" when gaps actually remain.
    gaps = p.get("gaps_after", [])
    ngaps = len(gaps)
    commanded_uncov = [z["fsa"] for z in (p.get("reasoning") or {}).get("zones", [])
                       if not z["covered_after"]]
    sample = commanded_uncov or gaps[:4]
    parts = [f"{p['n_moves']} relocations ({moves}).",
             f"Coverage {p['before_pct']*100:.0f}%->{p['after_pct']*100:.0f}%, "
             f"healed {len(p['gaps_healed'])} gaps."]
    if ngaps:
        ex = f" (e.g. {', '.join(sample)})" if sample else ""
        parts.append(f"{ngaps} zone(s) still uncovered{ex}.")
    else:
        parts.append("All demand zones are covered.")
    parts.extend(p.get("notes", []))
    return " ".join(parts)


def _explain_nim(payload: dict) -> str:
    system = (
        "You are an ambulance dispatch assistant. In 2 short sentences, summarize "
        "the relocation result for the operator: the overall coverage change and the "
        "number of zones still uncovered. State the uncovered count exactly as given "
        "in the facts; NEVER say 'no zones uncovered' unless the facts say all zones "
        "are covered. Do not enumerate every move. No preamble, no markdown."
    )
    return _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": _explain_facts(payload)},
        ],
        temperature=0.2,
        max_tokens=400,
    ).strip()


def _explain_template(p: dict) -> str:
    if p["status"] == "Infeasible":
        return (
            "No feasible relocation under the given constraints — the protected "
            "zones can't all be covered by the available units. Relax the "
            "constraints or add a unit."
        )
    if p["n_moves"] == 0:
        story = _coverage_story(p)
        story = (" " + story) if story else ""
        return (
            f"No moves needed: coverage is already at {p['before_pct']*100:.0f}% "
            f"of demand.{story} (solved in {p['solve_time_ms']:.0f} ms)"
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
    story = _coverage_story(p)
    story = (" " + story) if story else ""
    why = _decision_story(p)
    why = (" " + why) if why else ""
    return (
        f"Recommend {p['n_moves']} move(s): "
        + "; ".join(parts)
        + f". Demand coverage rises {p['before_pct']*100:.0f}% → "
        f"{p['after_pct']*100:.0f}%.{healed}{story}{why} "
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
