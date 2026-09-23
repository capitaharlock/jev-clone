"""Client for the Jev teacher API (#T-teacher-probe).

The teacher is the model this repo is cloning. It is the only honest ruler
for "how far are we": the release threshold in `#T-release-gate` is a number
WE wrote, the teacher's accuracy on OUR cut is not.

Three things this module refuses to do, because each one has already burned
a gate somewhere in this repo:

* **Hold the key.** The key is read from `JEV_TEACHER_API_KEY` or from a file
  outside git (`JEV_TEACHER_KEY_FILE`, default `.secrets/jev-api-key`). It is
  never written to a gate, a cache entry, a log line or a traceback —
  `redact()` is applied to everything that leaves this module.
* **Spend silently.** Every call goes through a `Budget`: a hard cap on calls
  and on dollars. Exceeding it raises `BudgetExceeded` instead of continuing,
  and the running cost is published by every gate that uses the client.
* **Pay twice for the same row.** Answers are cached on disk under
  `artifacts/cache/teacher/` keyed by sha256 of
  `(model, state, question, option ids and texts)`. Re-running an experiment
  costs zero, which is what makes `#T-teacher-kappa` free.

The wire format is the documented System-One shape::

    POST <endpoint>
    {"model": "...", "state": "...", "questions": {"<id>": {
        "type": "choice", "instructions": "...", "criteria": {"<opt>": "..."}}}}

and the answer is parsed defensively: providers differ on whether they return
`choice`/`answer`/`value` and whether the full distribution comes back at all.
A missing distribution is recorded as `probs: None`, never faked — a gate that
needs probabilities has to see that they were absent.

CLI:
    python3 -m eval.teacher ping        # one real call, prints choice + cost
    python3 -m eval.teacher config      # what is configured, key redacted
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ENDPOINT_ENV = "JEV_TEACHER_ENDPOINT"
MODEL_ENV = "JEV_TEACHER_MODEL"
KEY_ENV = "JEV_TEACHER_API_KEY"
KEY_FILE_ENV = "JEV_TEACHER_KEY_FILE"
PRICE_ENV = "JEV_TEACHER_INPUT_USD_PER_MTOK"

DEFAULT_MODEL = "jev-latest"
DEFAULT_KEY_FILE = ROOT / ".secrets" / "jev-api-key"
#: published list price at the time of writing; overridable by env because a
#: gate that hardcodes a price lies the day the price moves
DEFAULT_INPUT_USD_PER_MTOK = 0.042
CACHE_DIR = ROOT / "artifacts" / "cache" / "teacher"

RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.5


class TeacherError(RuntimeError):
    """Any failure that is the teacher's, not ours."""


class NotConfigured(TeacherError):
    """No endpoint or no key: the caller has to stop, not fall back."""


class BudgetExceeded(TeacherError):
    """The run asked for more calls or more dollars than it was granted."""


# -- configuration --------------------------------------------------------

def endpoint() -> str:
    return os.environ.get(ENDPOINT_ENV, "").strip()


def model() -> str:
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def key_file() -> Path:
    raw = os.environ.get(KEY_FILE_ENV, "").strip()
    return Path(raw) if raw else DEFAULT_KEY_FILE


def api_key() -> str:
    """The key, from env or from a file outside git. Never from the repo."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    path = key_file()
    if path.is_file():
        return path.read_text().strip()
    return ""


def input_price() -> float:
    raw = os.environ.get(PRICE_ENV, "").strip()
    try:
        return float(raw) if raw else DEFAULT_INPUT_USD_PER_MTOK
    except ValueError:
        return DEFAULT_INPUT_USD_PER_MTOK


def configured() -> bool:
    return bool(endpoint()) and bool(api_key())


def redact(text: str) -> str:
    """Strip the key out of anything about to be printed or serialised."""
    key = api_key()
    if key and key in text:
        text = text.replace(key, f"<key:{fingerprint()}>")
    return text


def fingerprint() -> str:
    """A stable 8-hex id of the key, safe to publish in a gate.

    Lets two artifacts say "same key" without either of them holding it.
    """
    key = api_key()
    if not key:
        return "none"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def config_card() -> dict:
    """What a gate publishes about how it was configured. No secrets."""
    return {
        "endpoint": endpoint() or None,
        "model": model(),
        "key_fingerprint": fingerprint(),
        "key_source": ("env" if os.environ.get(KEY_ENV, "").strip()
                       else (str(key_file()) if api_key() else None)),
        "input_usd_per_mtok": input_price(),
        "configured": configured(),
    }


# -- budget ---------------------------------------------------------------

class Budget:
    """A hard ceiling on what one run may spend.

    Both limits are checked BEFORE the call, so a run stops at the limit
    instead of one call past it.
    """

    def __init__(self, max_calls: int = 500, max_usd: float = 1.0) -> None:
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.calls = 0
        self.cached = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"call budget spent: {self.calls}/{self.max_calls}")
        if self.usd >= self.max_usd:
            raise BudgetExceeded(
                f"dollar budget spent: ${self.usd:.4f}/${self.max_usd:.4f}")

    def charge(self, input_tokens: int, output_tokens: int = 0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += input_tokens * input_price() / 1_000_000

    def card(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cached,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
            "max_calls": self.max_calls,
            "max_usd": self.max_usd,
            "usd_per_call": round(self.usd / self.calls, 8) if self.calls
            else 0.0,
        }


def estimate_tokens(payload: dict) -> int:
    """Fallback token count when the provider returns no `usage`.

    Deliberately crude (4 chars per token) and labelled as an estimate
    wherever it is published: an invented precise number is worse than an
    honest rough one.
    """
    return max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)


# -- the client -----------------------------------------------------------

class TeacherClient:
    """Disk-cached, budgeted, retrying client for one teacher endpoint."""

    def __init__(self, budget: Budget | None = None,
                 cache_dir: Path | None = None, timeout: int = 60,
                 log=print) -> None:
        self.budget = budget or Budget()
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.timeout = timeout
        self.log = log
        self.estimated_tokens = 0

    # -- cache ------------------------------------------------------------
    def cache_key(self, state: str, question: str, options: list) -> str:
        blob = json.dumps({
            "model": model(),
            "state": state,
            "question": question,
            "options": [[o.get("id", ""), o.get("text", "")]
                        for o in options],
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, key: str) -> dict | None:
        path = self.cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except ValueError:
            return None

    def _store(self, key: str, value: dict) -> None:
        path = self.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")

    # -- one decision -----------------------------------------------------
    def decide(self, state: str, question: str, options: list,
               question_id: str = "q") -> dict:
        """One forced choice over `options`, same shape our head is scored on.

        Returns `{"choice": <option id or None>, "probs": {...}|None,
        "cached": bool, "raw": <provider answer>}`.
        """
        key = self.cache_key(state, question, options)
        hit = self._cached(key)
        if hit is not None:
            self.budget.cached += 1
            return {**hit, "cached": True}
        if not configured():
            raise NotConfigured(
                f"set {ENDPOINT_ENV} and {KEY_ENV} (or put the key in "
                f"{key_file()}); nothing here is stubbed")
        self.budget.check()
        payload = self.build_payload(state, question, options, question_id)
        raw, usage = self._post(payload)
        answer = parse_answer(raw, question_id, options)
        used = int(usage.get("input_tokens") or 0)
        if not used:
            used = estimate_tokens(payload)
            self.estimated_tokens += used
            answer["tokens_estimated"] = True
        self.budget.charge(used, int(usage.get("output_tokens") or 0))
        answer["input_tokens"] = used
        self._store(key, answer)
        return {**answer, "cached": False}

    def build_payload(self, state: str, question: str, options: list,
                      question_id: str = "q") -> dict:
        return {
            "model": model(),
            "state": state,
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": question,
                    "criteria": {o["id"]: o.get("text") or o["id"]
                                 for o in options},
                },
            },
        }

    # -- transport --------------------------------------------------------
    def _post(self, payload: dict) -> tuple:
        body = json.dumps(payload, ensure_ascii=False).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key()}",
        }
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            req = urllib.request.Request(endpoint(), data=body,
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    doc = json.loads(r.read())
                return doc, usage_of(doc)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                last = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRY_STATUS:
                    raise TeacherError(redact(last)) from None
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS - 1:
                wait = BACKOFF_BASE ** attempt
                self.log(f"[teacher] {redact(last)}; retry in {wait:.1f}s")
                time.sleep(wait)
        raise TeacherError(redact(f"{MAX_ATTEMPTS} attempts failed: {last}"))


def usage_of(doc: dict) -> dict:
    """Token usage, whatever the provider decided to call it."""
    usage = doc.get("usage") or doc.get("token_usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": (usage.get("input_tokens")
                         or usage.get("prompt_tokens")
                         or usage.get("state_tokens") or 0),
        "output_tokens": (usage.get("output_tokens")
                          or usage.get("completion_tokens") or 0),
    }


def parse_answer(doc: dict, question_id: str, options: list) -> dict:
    """The chosen option id and, if the provider sent one, its distribution.

    An answer that names an option the request never offered is recorded as
    `off_menu` and counted as wrong — never snapped to the nearest option.
    """
    answers = doc.get("answers") or doc.get("decisions") or {}
    ans = answers.get(question_id) if isinstance(answers, dict) else None
    if ans is None and isinstance(answers, dict) and len(answers) == 1:
        ans = next(iter(answers.values()))
    if not isinstance(ans, dict):
        return {"choice": None, "probs": None, "confidence": None,
                "off_menu": False, "unparsed": True}
    choice = None
    for field in ("choice", "answer", "value", "id", "option", "label"):
        if isinstance(ans.get(field), str):
            choice = ans[field]
            break
    probs = None
    for field in ("probabilities", "distribution", "probs", "scores"):
        cand = ans.get(field)
        if isinstance(cand, dict) and cand:
            probs = {str(k): float(v) for k, v in cand.items()}
            break
    if choice is None and probs:
        choice = max(sorted(probs), key=lambda k: probs[k])
    ids = {o["id"] for o in options}
    return {
        "choice": choice,
        "probs": probs,
        "confidence": (float(ans["confidence"])
                       if isinstance(ans.get("confidence"), (int, float))
                       else None),
        "off_menu": bool(choice is not None and choice not in ids),
        "unparsed": choice is None,
    }


# -- CLI ------------------------------------------------------------------

def main(argv: list) -> int:
    cmd = argv[0] if argv else "config"
    if cmd == "config":
        print(json.dumps(config_card(), indent=2))
        return 0 if configured() else 1
    if cmd == "ping":
        client = TeacherClient(Budget(max_calls=1, max_usd=0.01))
        options = [{"id": "billing", "text": "Payments, invoicing, refunds"},
                   {"id": "technical", "text": "Bugs, outages, integrations"}]
        try:
            out = client.decide("Help! My payouts have failed for 3 days.",
                                "Which team should handle this?", options)
        except TeacherError as exc:
            print(f"FAIL: {exc}")
            return 1
        print(json.dumps({"answer": out, "budget": client.budget.card()},
                         indent=2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
