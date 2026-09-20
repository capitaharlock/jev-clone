"""Cheap teacher adapter (#T-teacher-intent): labeler + judge with cache.

Real JEv teacher via env (short prompts by design):
  JEV_TEACHER_URL / JEV_TEACHER_KEY  (OpenAI-compatible /v1/chat/completions)
  JEV_TEACHER_EUR_PER_1K             (default 0.002 = promos cortos baratos)

Without env configured the client runs the deterministic local stub
(seed-fixed, ~90% gold agreement when gold_hint is given) so the pilot
exercises the full plumbing at zero cost. The stub is NEVER presented
as the external teacher: records carry backend + billable flags.

Cache in artifacts/data-raw/teacher/labels.json: a second pass over the
same texts costs 0 calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artifacts" / "data-raw" / "teacher" / "labels.json"

VERSION = "teacher-client-v1"
EUR_PER_1K = float(os.environ.get("JEV_TEACHER_EUR_PER_1K", "0.002"))

KEYWORDS: dict[str, list[str]] = {
    # Qwen-proxy heuristic: cheap, deterministic, documented as proxy.
    "spam": ["ganaste", "clic", "gratis", "préstamo pre-aprobado", "suspendida",
             "herencia", "felicidades", "gana ", "sin esfuerzo", "regala"],
    "urgente": ["urgente", "urgent", "hoy", "ahora", "bloqueada", "corte de",
                "facturado", "check-in", "checked in", "fuga", "termina hoy",
                "plazo", "3 horas", "3 hours"],
    "responder": ["revisar", "review", "confirma", "confirm", "propuesta",
                  "invitación", "invitation", "presupuesto", "entrevista",
                  "interview", "opini", "duda", "pregunta"],
}


def _key(text: str, backend: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(f"{VERSION}:{backend}:{norm}".encode()).hexdigest()[:32]


def _heuristic(text: str) -> tuple[str, float]:
    t = text.lower()
    for label, words in KEYWORDS.items():
        if any(w in t for w in words):
            return label, 0.7
    return "archivar", 0.6


class TeacherClient:
    def __init__(self, backend: str = "stub", seed: int = 0,
                 endpoint: str | None = None, api_key: str | None = None,
                 cache_path: Path = CACHE):
        self.backend = backend
        self.seed = seed
        self.endpoint = endpoint or os.environ.get("JEV_TEACHER_URL", "")
        self.api_key = api_key or os.environ.get("JEV_TEACHER_KEY", "")
        self.cache_path = Path(cache_path)
        self.calls_made = 0       # non-cached predictions
        self.billable_calls = 0   # HTTP calls to the external teacher
        self.cache_hits = 0
        try:
            self._cache = json.loads(self.cache_path.read_text())
        except (FileNotFoundError, ValueError):
            self._cache = {}

    def stats(self) -> dict:
        total = self.calls_made + self.cache_hits
        eur = self.billable_calls / 1000 * EUR_PER_1K
        return {"backend": self.backend, "total": total,
                "calls_made": self.calls_made, "cache_hits": self.cache_hits,
                "billable_calls": self.billable_calls,
                "cost_eur": round(eur, 4),
                "cost_eur_per_1k": EUR_PER_1K,
                "endpoint_configured": bool(self.endpoint)}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False))

    def _http_label(self, text: str) -> tuple[str, float] | None:
        prompt = ("Clasifica este email en UNA palabra: archivar, responder, "
                  f"urgente o spam. Solo la palabra.\n{text[:800]}")
        body = json.dumps({"model": "jev",
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 5, "temperature": 0}).encode()
        try:
            req = urllib.request.Request(
                self.endpoint.rstrip("/") + "/v1/chat/completions", data=body,
                headers={"content-type": "application/json",
                         "authorization": f"Bearer {self.api_key}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=20) as r:
                out = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            lab = out.strip().lower().split()[0].strip(".,;:")
            if lab not in ("archivar", "responder", "urgente", "spam"):
                return None
            return lab, 0.9
        except Exception:
            return None

    def _stub_label(self, text: str, gold: str | None, i: int) -> tuple[str, float]:
        if gold is not None:
            # Deterministic noise: ~90% gold, else runner-up by keyword.
            rng = random.Random(f"{self.seed}:{text[:64]}:{i}")
            if rng.random() < 0.9:
                return gold, 0.9
        lab, conf = _heuristic(text)
        if gold is not None and lab != gold:
            return lab, conf
        return (gold, 0.75) if gold else (lab, conf)

    def label(self, texts: list[str],
              gold_hint: list[str] | None = None) -> list[dict]:
        out = []
        for i, text in enumerate(texts):
            k = _key(text, self.backend)
            if k in self._cache:
                self.cache_hits += 1
                out.append({**self._cache[k], "cached": True})
                continue
            rec: dict
            if self.endpoint:
                got = self._http_label(text)
                if got is not None:
                    self.billable_calls += 1
                    rec = {"label": got[0], "conf": got[1],
                           "backend": "jev-external", "billable": True}
                else:
                    lab, conf = self._stub_label(
                        text, gold_hint[i] if gold_hint else None, i)
                    rec = {"label": lab, "conf": conf,
                           "backend": "stub-http-fallback", "billable": False}
            else:
                lab, conf = self._stub_label(
                    text, gold_hint[i] if gold_hint else None, i)
                rec = {"label": lab, "conf": conf, "backend": "stub-local",
                       "billable": False}
            self.calls_made += 1
            self._cache[k] = rec
            out.append({**rec, "cached": False})
        self._save()
        return out

    def qwen_proxy(self, texts: list[str]) -> list[dict]:
        """Documented proxy for Qwen-local labels (keyword heuristic).

        A full 500-row Ollama pass is too slow for the pilot; forward
        testing in the tester engine already judges with real Qwen.
        """
        return [{"label": lab, "conf": conf, "backend": "qwen-proxy",
                 "billable": False}
                for lab, conf in (_heuristic(t) for t in texts)]


def agreement_report(a: list[str], b: list[str]) -> dict:
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    return {"n": n, "agree": agree,
            "agreement": round(agree / n, 4) if n else 0.0}
