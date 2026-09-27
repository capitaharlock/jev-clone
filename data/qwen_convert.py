"""Qwen converter: enrich prefetched training sets nonstop (#T-prefetch).

Watches `artifacts/data-prefetch/*.jsonl` (universal schema rows + P1 raw
dumps) and uses the local Qwen (`qwen3.8:27b-mlx` via ollama,
http://localhost:11434) to produce one augmented variant per row:
paraphrase + quality flags, appended to `artifacts/data-qwen/<id>.aug.jsonl`.

Runs FOREVER (poll loop): resumable via `_offsets.json`, skips rows already
done, picks up new files as the prefetch lands them. Never touches
eval-only CLINC150 for train — its snapshot has no converter output.
Kill with pkill -f qwen_convert.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(BASE, "artifacts", "data-prefetch")
OUT_DIR = os.path.join(BASE, "artifacts", "data-qwen")
OFFSETS = os.path.join(OUT_DIR, "_offsets.json")
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8:27b-mlx")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
POLL_SECS = int(os.environ.get("QWEN_POLL_SECS", "30"))

# Eval-only fence: never augment these into train.
SKIP = ("clinc150",)

SYSTEM = (
    "You rewrite one training example as a JSON object with keys "
    "'text' (paraphrase preserving meaning and label), 'label' (copied "
    "verbatim), 'quality' (good/noisy). Reply with ONLY that JSON, no prose."
)


def chat(row: dict) -> dict | None:
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "think": False,  # reasoning eats the token budget, content comes back empty
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(row, ensure_ascii=False)[:2000]},
        ],
        "options": {"temperature": 0.7, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())["message"]["content"].strip()
        # thinking models wrap reasoning in <think>…</think>; prose ok
        if "</think>" in out:
            out = out.split("</think>", 1)[1]
        if out.startswith("```"):
            out = out.split("\n", 1)[1].rsplit("```", 1)[0]
        start, end = out.find("{"), out.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON in reply: {out[:120]!r}")
        aug = json.loads(out[start:end + 1])
        return {"src": row, "aug": aug, "model": MODEL}
    except Exception as e:
        print(f"[qwen] row failed: {e}", flush=True)
        return None


def load_offsets() -> dict:
    try:
        with open(OFFSETS) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[qwen] converter up model={MODEL} in={IN_DIR}", flush=True)
    while True:
        offsets = load_offsets()
        files = sorted(glob.glob(os.path.join(IN_DIR, "*.jsonl")))
        if not files:
            print("[qwen] no prefetch files yet, waiting…", flush=True)
        for path in files:
            fid = os.path.basename(path)[:-len(".jsonl")]
            if fid in SKIP:
                continue
            done = offsets.get(fid, 0)
            out_path = os.path.join(OUT_DIR, f"{fid}.aug.jsonl")
            n_new = 0
            try:
                with open(path) as f:
                    for i, line in enumerate(f):
                        if i < done:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            done = i + 1
                            continue
                        rec = chat(row)
                        if rec is None:
                            break  # ollama down; retry next pass
                        with open(out_path, "a") as o:
                            o.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done = i + 1
                        n_new += 1
                        if n_new % 25 == 0:
                            offsets[fid] = done
                            with open(OFFSETS, "w") as of:
                                json.dump(offsets, of)
                            print(f"[qwen] {fid}: +{n_new} (offset {done})",
                                  flush=True)
            except FileNotFoundError:
                continue
            offsets[fid] = done
            with open(OFFSETS, "w") as of:
                json.dump(offsets, of)
            if n_new:
                print(f"[qwen] {fid}: pass done +{n_new} (offset {done})",
                      flush=True)
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    sys.exit(main())
