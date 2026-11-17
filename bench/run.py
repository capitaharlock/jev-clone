"""Score one system on the frozen benchmark and write a result file.

Every system answers the same rows and gets the same metrics: forced
accuracy over the offered products, the 95% Wilson lower bound, a
breakdown by question shape, and latency per decision.

    python -m bench.run own --model checkpoints/asimovia-decision-v1
    python -m bench.run rules
    python -m bench.run tinyjev --model path/to/TinyJev-0.6B
    python -m bench.run ollama-decision --model nimble
    python -m bench.run ollama-chat --model qwen3.8:27b --limit 600

Systems:
    own              this repository's checkpoint, in process
    rules            the hand-written parser in bench/rules.py (the ceiling)
    tinyjev          a TinyJev pointer checkpoint through the `tinyjev` package
    ollama-decision  any Ollama decision model, through POST /v1/systemone
    ollama-chat      any Ollama chat model, zero-shot, answering one letter
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "product_specs.dev.jsonl"
RESULTS = ROOT / "bench" / "results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified(rows: list[dict], limit: int) -> list[dict]:
    """Every n-th row of each (difficulty, K, language) cell, in file order."""
    if not limit or limit >= len(rows):
        return rows
    cells = defaultdict(list)
    for row in rows:
        cells[(row["difficulty"], len(row["candidates"]), row["lang"])].append(row)
    keep = set()
    for members in cells.values():
        share = max(1, round(len(members) * limit / len(rows)))
        step = len(members) / share
        keep.update(id(members[int(i * step)]) for i in range(share))
    return [row for row in rows if id(row) in keep]


def wilson_lower(hits: int, n: int, z: float = 1.959964) -> float:
    if not n:
        return 0.0
    p = hits / n
    centre = p + z * z / (2 * n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, (centre - half) / (1 + z * z / n))


def post(url: str, payload: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


class Own:
    def __init__(self, args):
        import torch
        from model.checkpoint import load_model
        from model.inference import predict
        self.torch, self.predict = torch, predict
        self.device = torch.device(args.device)
        self.model = load_model(args.model, self.device)
        lineage = self.model.lineage()
        self.info = {"name": "asimovia-decision-v1", "kind": "own",
                     "parameters": lineage["parameters"],
                     "checkpoint": str(Path(args.model).name),
                     "device": args.device, "torch": torch.__version__}

    def sync(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def answer(self, row: dict) -> str | None:
        return self.predict(self.model, [row], self.device)[0]["choice"]

    def throughput(self, rows: list[dict]) -> float:
        started = time.perf_counter()
        self.predict(self.model, rows, self.device, batch_size=32)
        self.sync()
        return len(rows) / (time.perf_counter() - started)


class Rules:
    def __init__(self, args):
        from bench.rules import decide
        self.decide = decide
        self.info = {"name": "rules", "kind": "program", "parameters": 0}

    def answer(self, row: dict) -> str | None:
        return self.decide(row)


class TinyJev:
    def __init__(self, args):
        import tinyjev
        self.agent = tinyjev.load(str(args.model), backend="torch",
                                  device=args.device)
        manifest_path = Path(args.model) / "tinyjev.json"
        manifest = (json.loads(manifest_path.read_text())
                    if manifest_path.is_file() else {})
        self.info = {"name": manifest.get("name") or Path(args.model).name,
                     "kind": "tinyjev", "revision": Path(args.model).name,
                     "tinyjev": tinyjev.__version__, "device": args.device,
                     "manifest": manifest}

    def answer(self, row: dict) -> str | None:
        reply = self.agent.predict(system_one(row))
        return choice_of(reply)


def system_one(row: dict) -> dict:
    """The benchmark row as a System One choice question."""
    return {"state": row["state"], "questions": {"pick": {
        "type": "choice", "instructions": row["question"],
        "criteria": {c["id"]: c["text"] for c in row["candidates"]}}}}


def choice_of(reply: dict) -> str | None:
    """The picked id from a System One reply; tinyjev nests it per state."""
    answers = reply.get("answers")
    if answers is None and reply.get("states"):
        answers = reply["states"][0].get("answers")
    answer = (answers or {}).get("pick") or {}
    if answer.get("choice"):
        return answer["choice"]
    probabilities = answer.get("probabilities") or {}
    return max(probabilities, key=probabilities.get) if probabilities else None


class OllamaDecision:
    def __init__(self, args):
        self.url, self.model = args.ollama.rstrip("/"), args.model
        version = get(f"{self.url}/api/version").get("version")
        show = post(f"{self.url}/api/show", {"model": self.model})
        self.info = {"name": self.model, "kind": "ollama-decision",
                     "ollama": version, "digest": digest(self.url, self.model),
                     "details": show.get("details")}

    def answer(self, row: dict) -> str | None:
        return choice_of(post(f"{self.url}/v1/systemone",
                              {"model": self.model, **system_one(row)}))


class OllamaChat:
    def __init__(self, args):
        self.url, self.model = args.ollama.rstrip("/"), args.model
        version = get(f"{self.url}/api/version").get("version")
        show = post(f"{self.url}/api/show", {"model": self.model})
        self.info = {"name": self.model, "kind": "ollama-chat",
                     "ollama": version, "digest": digest(self.url, self.model),
                     "details": show.get("details"), "prompt": "zero-shot"}

    def answer(self, row: dict) -> str | None:
        options = "\n".join(f"{LETTERS[i]}. {c['text']}"
                            for i, c in enumerate(row["candidates"]))
        prompt = (f"{row['state']}\n\n{row['question']}\n\n{options}\n\n"
                  "Answer with the letter of one option and nothing else.")
        reply = post(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": 8}})
        text = (reply.get("message") or {}).get("content", "")
        match = re.search(r"\b([A-Z])\b", text.strip().upper())
        if not match or LETTERS.index(match.group(1)) >= len(row["candidates"]):
            return None
        return row["candidates"][LETTERS.index(match.group(1))]["id"]


def digest(url: str, model: str) -> str | None:
    for item in get(f"{url}/api/tags").get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return item.get("digest")
    return None


SYSTEMS = {"own": Own, "rules": Rules, "tinyjev": TinyJev,
           "ollama-decision": OllamaDecision, "ollama-chat": OllamaChat}


def cell_keys(row: dict) -> list[str]:
    salt = any("sodium" in c["text"] or "sodio" in c["text"]
               for c in row["candidates"])
    return [f"difficulty:{row['difficulty'].split('+')[0]}",
            f"options:{len(row['candidates'])}", f"lang:{row['lang']}",
            f"units:{'sodium_mg' if salt else 'salt_g'}"]


def score(system, rows: list[dict], progress: bool = True) -> dict:
    hits = decidable_hits = decidable = unparsed = 0
    cells = defaultdict(lambda: [0, 0])
    latencies = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        choice = system.answer(row)
        if hasattr(system, "sync"):
            system.sync()
        latencies.append((time.perf_counter() - started) * 1000)
        good = choice == row["answer"]
        unparsed += choice is None
        hits += good
        if row.get("decidable", True):
            decidable += 1
            decidable_hits += good
        for key in cell_keys(row):
            cells[key][0] += good
            cells[key][1] += 1
        if progress and (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(rows)} accuracy {hits / (index + 1):.4f}",
                  file=sys.stderr, flush=True)
    latencies.sort()
    return {
        "rows": len(rows),
        "hits": hits,
        "accuracy": hits / len(rows),
        "wilson_lower": wilson_lower(hits, len(rows)),
        "decidable_rows": decidable,
        "decidable_accuracy": decidable_hits / decidable if decidable else None,
        "chance": sum(1 / len(r["candidates"]) for r in rows) / len(rows),
        "unparsed": unparsed,
        "by": {key: {"accuracy": h / n, "rows": n}
               for key, (h, n) in sorted(cells.items())},
        "latency_ms": {"p50": statistics.median(latencies),
                       "p95": latencies[int(0.95 * (len(latencies) - 1))],
                       "mean": statistics.fmean(latencies)},
    }


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    cli.add_argument("system", choices=sorted(SYSTEMS))
    cli.add_argument("--model", default=str(ROOT / "checkpoints"
                                            / "asimovia-decision-v1"))
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--ollama", default="http://localhost:11434")
    cli.add_argument("--benchmark", type=Path, default=BENCHMARK)
    cli.add_argument("--limit", type=int, default=0,
                     help="stratified subset size for slow systems")
    cli.add_argument("--out", type=Path)
    args = cli.parse_args(argv)
    rows = stratified(read_rows(args.benchmark), args.limit)
    system = SYSTEMS[args.system](args)
    result = {"system": system.info, "benchmark": args.benchmark.name,
              "subset": args.limit or None,
              "host": {"machine": platform.machine(),
                       "platform": platform.platform(),
                       "python": platform.python_version()},
              "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
              **score(system, rows)}
    if hasattr(system, "throughput"):
        result["batched_decisions_per_s"] = system.throughput(rows)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", system.info["name"])
    where = f"--{args.device}" if "device" in system.info else ""
    subset = f"--subset{args.limit}" if args.limit else ""
    out = args.out or RESULTS / f"{args.system}--{safe}{where}{subset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("accuracy", "decidable_accuracy",
                                             "rows", "latency_ms")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
