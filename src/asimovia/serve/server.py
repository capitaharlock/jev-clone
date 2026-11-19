"""A small HTTP server for the checkpoint, built on the standard library.

GET  /                 the examples page
GET  /examples         every sample request in examples/
GET  /v1/health        model format, parameters, device
POST /v1/decide        {"state", "question", "candidates": [{"id", "text"}]}

python -m asimovia.serve --model checkpoints/asimovia-decision-v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import torch

from ..model.checkpoint import load_model
from ..model.inference import predict

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path("examples")
MAX_BODY = 1 << 20
MAX_OPTIONS = 26


def check_request(body: dict) -> str | None:
    """The first problem with a decide request, or None when it is usable."""
    for field in ("state", "question"):
        if not isinstance(body.get(field), str) or not body[field].strip():
            return f"{field} must be non-empty text"
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= MAX_OPTIONS:
        return f"candidates must hold 2 to {MAX_OPTIONS} options"
    ids = []
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("text"), str)
            or not candidate["text"].strip()
        ):
            return "every candidate needs a non-empty text"
        ids.append(str(candidate.get("id", "")))
    if "" in ids or len(set(ids)) != len(ids):
        return "candidate ids must be present and unique"
    return None


class Decider:
    """Holds one model; a lock keeps concurrent requests off the same GPU."""

    def __init__(self, folder: Path, device: str):
        self.device = torch.device(device)
        self.model = load_model(folder, self.device)
        self.lock = Lock()
        self.info = {**self.model.lineage(), "device": str(self.device), "checkpoint": folder.name}

    def decide(self, body: dict) -> dict:
        row = {
            "state": body["state"],
            "question": body["question"],
            "candidates": [{"id": str(c["id"]), "text": c["text"]} for c in body["candidates"]],
        }
        with self.lock:
            started = time.perf_counter()
            result = predict(self.model, [row], self.device)[0]
            if self.device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
        return {**result, "latency_ms": round(elapsed * 1000, 1), "model": self.info["format"]}


def examples(folder: Path = EXAMPLES) -> list[dict]:
    return [
        {"name": path.stem, **json.loads(path.read_text(encoding="utf-8"))}
        for path in sorted(Path(folder).glob("*.json"))
    ]


def handler(decider: Decider, samples: Path = EXAMPLES) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "asimovia/1"

        def log_message(self, format: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

        def send(self, status: int, payload: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, status: int, payload: dict | list) -> None:
            self.send(
                status,
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self.send(
                    HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8"
                )
            elif path == "/examples":
                self.json(HTTPStatus.OK, examples(samples))
            elif path == "/v1/health":
                self.json(HTTPStatus.OK, decider.info)
            else:
                self.json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/v1/decide":
                self.json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY:
                self.json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": f"body must be 1 to {MAX_BODY} bytes"},
                )
                return
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                self.json(HTTPStatus.BAD_REQUEST, {"error": "body is not JSON"})
                return
            problem = (
                check_request(body) if isinstance(body, dict) else "body must be a JSON object"
            )
            if problem:
                self.json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": problem})
                return
            self.json(HTTPStatus.OK, decider.decide(body))

    return Handler


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("--model", type=Path, default=Path("checkpoints/asimovia-decision-v1"))
    cli.add_argument("--examples", type=Path, default=EXAMPLES)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--device", default="cpu", help="cpu, mps or cuda; cpu answers in tens of ms")
    args = cli.parse_args(argv)
    decider = Decider(args.model, args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler(decider, args.examples))
    print(f"serving {decider.info['format']} on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
