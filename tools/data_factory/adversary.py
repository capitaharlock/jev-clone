"""Adversary: distractor + ambiguity variants (Qwen-value roles: descriptions,
paraphrases, distractors, ambiguity). Every variant keeps parent_example_id."""
from __future__ import annotations

from .teachers_local import LocalTeacher


def adversarial_variants(state: str, question: dict, key: str) -> list[dict]:
    t = LocalTeacher("qwen")
    variants = []
    # distractor option injection
    q2 = {**question, "options": list(question["options"]) + [
        {"id": "distractor", "text": t.distractor(key + ":d")}]}
    variants.append({"kind": "distractor", "question": q2})
    # paraphrased state ambiguity probe (same gold)
    variants.append({
        "kind": "ambiguity",
        "state": t.paraphrase(state, key + ":a"),
        "question": question,
    })
    return variants
