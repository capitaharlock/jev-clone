"""Env-gated external proposer for schema records (#T-gen-schemas).

The full code path exists: prompt, HTTP call, parse, validate. What is gated
is the *endpoint*. With ``JEV_GEN_ENDPOINT`` unset (the default, and the CI
case) the client logs ``STUB: would call <model> @ <endpoint>`` once per
(domain, language) and returns nothing, so the generator falls through to its
deterministic local sampler.

Only the *records* (which value each field takes) come from outside. Schema
assembly, the quota and the dedup are never stubbed: an external model cannot
talk the generator into an over-budget or duplicate schema.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .domains import Domain

ENDPOINT_ENV = "JEV_GEN_ENDPOINT"
MODEL_ENV = "JEV_GEN_MODEL"
KEY_ENV = "JEV_GEN_API_KEY"
DEFAULT_MODEL = "qwen3.6:27b-mlx"


def endpoint() -> str:
    return os.environ.get(ENDPOINT_ENV, "").strip()


def model() -> str:
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def configured() -> bool:
    return bool(endpoint())


def build_prompt(dom: Domain, lang: str, n: int) -> str:
    fields = {f.name: [k for k, _ in f.values()] for f in dom.fields}
    return (
        f"You are building {n} decision cases for the domain {dom.id!r} in "
        f"language {lang!r}. Answer ONLY with a JSON array of {n} objects. "
        f"Each object assigns exactly one value to every field, picked from "
        f"its allowed list, and the combination must be operationally "
        f"plausible (not uniform noise).\n"
        f"Allowed values: {json.dumps(fields, ensure_ascii=False)}"
    )


class RecordProposer:
    """Proposes field assignments. Stubbed unless an endpoint is configured."""

    def __init__(self, log=print, timeout: int = 60) -> None:
        self.log = log
        self.timeout = timeout
        self.n_calls = 0
        self.n_stubbed = 0
        self.n_records = 0
        self.n_dropped = 0
        self._announced: set[tuple[str, str]] = set()

    def propose(self, dom: Domain, lang: str, n: int) -> list[dict[str, str]]:
        if not configured():
            if (dom.id, lang) not in self._announced:
                self._announced.add((dom.id, lang))
                self.log(f"STUB: would call {model()} @ "
                         f"<{ENDPOINT_ENV} unset> for {n} record(s) of "
                         f"{dom.id}/{lang}; using the local generator")
            self.n_stubbed += 1
            return []
        self.n_calls += 1
        try:
            raw = self._post(build_prompt(dom, lang, n))
        except Exception as exc:  # network, auth, timeout — never fatal
            self.log(f"[gen-schemas] proposer failed ({exc}); local fallback")
            return []
        return self._parse(raw, dom)

    # --- internals --------------------------------------------------------
    def _post(self, prompt: str) -> str:
        body = json.dumps({
            "model": model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False,
        }).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(KEY_ENV, "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(endpoint(), data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read())
        choices = payload.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return payload.get("response", "")

    def _parse(self, raw: str, dom: Domain) -> list[dict[str, str]]:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            self.log("[gen-schemas] proposer returned no JSON array; local fallback")
            return []
        try:
            items = json.loads(raw[start:end + 1])
        except ValueError as exc:
            self.log(f"[gen-schemas] proposer JSON invalid ({exc}); local fallback")
            return []
        allowed = {f.name: {k for k, _ in f.values()} for f in dom.fields}
        out: list[dict[str, str]] = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                self.n_dropped += 1
                continue
            rec = {name: it.get(name) for name in allowed}
            if any(rec[name] not in allowed[name] for name in allowed):
                self.n_dropped += 1  # off-vocabulary assignment: dropped, never coerced
                continue
            out.append(rec)
        self.n_records += len(out)
        return out

    def report(self) -> dict:
        return {
            "configured": configured(),
            "endpoint_env": ENDPOINT_ENV,
            "model": model(),
            "calls": self.n_calls,
            "stubbed_calls": self.n_stubbed,
            "records_accepted": self.n_records,
            "records_dropped": self.n_dropped,
        }
