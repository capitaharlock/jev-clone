"""Dataset registry with hashes + license fence (#T-data-schema).

Every training artifact must reference a manifest produced here; a card
without immutable revision or SHA-256 cannot be registered, and
`train` usage requires an approved license/provenance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

USAGES = ("train", "research-only", "eval-only")

# Licenses approved for `train` use (mirrors source-register.md).
TRAIN_APPROVED_LICENSES = frozenset({"MIT", "Apache-2.0", "CC0", "CC0-1.0", "CC-BY-4.0"})

# Some datasets need extra provenance conditions before training.
TRAIN_GATED = {
    # HuffPost: mirror says CC0 but underlying content rights need own fence.
    "huffpost": "needs HuffPost content-rights fence decision (plan §148)",
    "boolq": "share-alike obligations (CC-BY-SA-3.0) must be documented",
}


@dataclass
class DatasetCard:
    id: str
    source_original: str
    mirror: str
    license: str
    usage: str  # train | research-only | eval-only
    revision: str  # immutable revision / commit; never a bare live URL
    sha256: str
    transform: str


class Registry:
    def __init__(self) -> None:
        self._cards: dict[str, DatasetCard] = {}

    def register(self, card: DatasetCard) -> None:
        errors = self.check_card(card)
        if errors:
            raise ValueError("; ".join(errors))
        self._cards[card.id] = card

    @staticmethod
    def check_card(card: DatasetCard) -> list[str]:
        errors: list[str] = []
        if card.usage not in USAGES:
            errors.append(f"usage must be one of {USAGES}")
        if not card.revision or not card.revision.strip():
            errors.append("revision is required (immutable, not a live URL)")
        if not card.sha256 or len(card.sha256) != 64:
            errors.append("sha256 (64 hex chars) is required")
        else:
            try:
                int(card.sha256, 16)
            except ValueError:
                errors.append("sha256 must be hex")
        if not card.source_original:
            errors.append("source_original is required")
        if not card.license:
            errors.append("license is required")
        return errors

    def get(self, id: str) -> DatasetCard:
        return self._cards[id]

    def train_ok(self, id: str) -> tuple[bool, str]:
        """License fence: (allowed, reason) for training on dataset id."""
        card = self.get(id)
        if card.usage != "train":
            return False, f"usage is {card.usage!r}, not train"
        if card.license not in TRAIN_APPROVED_LICENSES:
            return False, f"license {card.license!r} not approved for train"
        if card.id in TRAIN_GATED:
            return False, f"gated: {TRAIN_GATED[card.id]}"
        return True, "ok"

    def check_mix(self, ids: list[str]) -> list[str]:
        """Reject mixes that leak eval-only/research-only into train."""
        errors: list[str] = []
        for i in ids:
            ok, reason = self.train_ok(i)
            if not ok:
                errors.append(f"{i}: {reason}")
        return errors

    def manifest(self, ids: list[str]) -> dict:
        """Training manifest: pins every dataset revision + hash."""
        entries = [asdict(self.get(i)) for i in ids]
        blob = json.dumps(entries, sort_keys=True).encode()
        return {
            "datasets": entries,
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        }
