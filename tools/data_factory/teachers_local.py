"""Local deterministic mock teachers + HTTP client for real endpoints.

Default is offline/mocked so the pilot is reproducible. A real OpenAI-
compatible endpoint can be configured via JEV_TEACHER_ENDPOINT; the HTTP
path is only used when explicitly configured (never a UI).
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request


def _h(seed: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest(), 16)


PARAPHRASES = [
    "En otras palabras: {t}",
    "Dicho de otro modo: {t}",
    "Reformulación: {t}",
    "Versión alternativa: {t}",
]

DISTRACTORS = [
    "Ninguna de las anteriores es correcta.",
    "No hay información suficiente en el texto.",
    "Todo lo anterior es correcto.",
    "Solo bajo condiciones no mencionadas.",
    "La pregunta está mal planteada.",
]


class LocalTeacher:
    """Deterministic template teacher identified by name (Qwen/DeepSeek roles)."""

    def __init__(self, name: str):
        self.name = name

    def propose(self, state: str, key: str, k: int = 4) -> dict:
        """Propose a boolean/choice question grounded in state. No CoT."""
        h = _h(key, f"propose:{self.name}")
        kind = "boolean" if h % 3 == 0 else "choice"
        if kind == "boolean":
            options = [
                {"id": "yes", "text": "Sí"},
                {"id": "no", "text": "No"},
            ]
            gold = "yes" if (h >> 8) % 2 == 0 else "no"
        else:
            k = max(2, min(k, 6))
            options = [
                {"id": f"opt{i}", "text": f"Opción {i} sobre: {state[:48].strip()}"}
                for i in range(k)
            ]
            gold = options[(h >> 8) % k]["id"]
        return {
            "kind": kind,
            "question": f"Según el texto, ¿qué afirmación es correcta? [{key[:8]}]",
            "options": options,
            "gold": gold,
        }

    def solve(self, state: str, question: dict, key: str) -> str:
        h = _h(key, f"solve:{self.name}:{json.dumps(question, sort_keys=True)}")
        return question["gold"] if h % 10 < 8 else [
            o["id"] for o in question["options"] if o["id"] != question["gold"]
        ][(h >> 4) % (len(question["options"]) - 1)]

    def paraphrase(self, text: str, key: str) -> str:
        h = _h(key, f"para:{self.name}")
        return PARAPHRASES[h % len(PARAPHRASES)].format(t=text)

    def distractor(self, key: str) -> str:
        return DISTRACTORS[_h(key, f"dist:{self.name}") % len(DISTRACTORS)]


class HttpTeacher:
    """Thin OpenAI-compatible chat client; only used when endpoint configured."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            self.endpoint + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def teacher_endpoint_configured() -> bool:
    return bool(os.environ.get("JEV_TEACHER_ENDPOINT"))
