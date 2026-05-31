"""Head-to-head: ollama vs llama-server on the REAL parse/explain workload.

Uses the same OpenAI-compatible /v1/chat/completions call the app makes
(think:False, temperature 0, max_tokens 400) with a representative dispatch
prompt incl. the full FSA glossary, so prompt-eval cost is realistic.
Reports prompt tokens, gen tokens, wall-clock, and tok/s for each backend.
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
from load_world import load_world  # noqa: E402

world = load_world(os.path.join(ROOT, "datasets", "world.npz"),
                   os.path.join(ROOT, "datasets", "world_meta.json"))
FSA = list(world["fsa_index"])

SYSTEM = (
    "detailed thinking off\n"
    "You are an ambulance dispatch supervisor. Classify the operator's command "
    "and return ONLY a JSON object with keys intent, dispatch, constraints.\n"
    "- 'emergency': active incident, fill dispatch {zone, n_units, reason, priority}.\n"
    "- 'coverage': readiness request, fill constraints.\n"
    "Resolve FSA postal codes using this glossary:\n" + json.dumps({"fsa_codes": FSA})
)
USER = "Major collision in M5V, roll three units now."

BACKENDS = [
    ("ollama        ", "http://localhost:11434/v1", "nemotron3:33b"),
    ("llama-server  ", "http://127.0.0.1:8088/v1", "local-gguf"),
]


def hit(base, model, n_warmup=1, n_runs=3):
    body = lambda: json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER}],
        "temperature": 0.0, "max_tokens": 400, "think": False,
    }).encode()

    def one():
        t0 = time.perf_counter()
        req = urllib.request.Request(base + "/chat/completions", data=body(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        dt = time.perf_counter() - t0
        u = data.get("usage", {})
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), content

    for _ in range(n_warmup):
        one()
    runs = [one() for _ in range(n_runs)]
    dt = sum(r[0] for r in runs) / len(runs)
    pt = runs[-1][1]
    ct = sum(r[2] for r in runs) / len(runs)
    return dt, pt, ct, runs[-1][3]


print(f"FSA glossary size: {len(FSA)} codes\n")
for name, base, model in BACKENDS:
    try:
        dt, pt, ct, content = hit(base, model)
        tps = ct / dt if dt else 0
        print(f"{name} | wall {dt*1000:7.0f} ms | prompt {pt:4d} tok | "
              f"gen {ct:5.1f} tok | {tps:6.1f} tok/s")
        print(f"   -> {content[:200].strip()!r}\n")
    except Exception as e:
        print(f"{name} | ERROR: {e}\n")
