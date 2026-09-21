"""Per-domain budget for the decision-schema generator (#T-gen-schemas).

The loop is bounded by a QUOTA, not by a rate: a cell is
(domain, language, K-cardinality) and the loop ends when every cell is full.
"358 rows/min indefinitely" is precisely the failure mode this replaces —
past the first couple of thousand rows that rate added cost and no
information (audit-2026-09-21, finding C).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .domains import DOMAIN_IDS, MAX_K, MIN_K
from .vocab import LANGS

Cell = tuple[str, str, int]  # (domain, language, k)


@dataclass
class Budget:
    quota: dict[Cell, int]
    filled: dict[Cell, int] = field(default_factory=dict)

    @classmethod
    def plan(cls, per_cell: int = 8, domains: tuple[str, ...] = DOMAIN_IDS,
             languages: tuple[str, ...] = LANGS,
             ks: tuple[int, ...] | None = None) -> "Budget":
        ks = ks if ks is not None else tuple(range(MIN_K, MAX_K + 1))
        for k in ks:
            if not MIN_K <= k <= MAX_K:
                raise ValueError(f"K={k} outside [{MIN_K}, {MAX_K}]")
        return cls({(d, lg, k): per_cell
                    for d in domains for lg in languages for k in ks})

    # --- accounting -------------------------------------------------------
    def cells(self) -> list[Cell]:
        return list(self.quota)

    def total(self) -> int:
        return sum(self.quota.values())

    def taken(self) -> int:
        return sum(self.filled.values())

    def room(self, cell: Cell) -> int:
        return self.quota.get(cell, 0) - self.filled.get(cell, 0)

    def is_full(self, cell: Cell) -> bool:
        return self.room(cell) <= 0

    def full(self) -> bool:
        return all(self.is_full(c) for c in self.quota)

    def open_cells(self) -> list[Cell]:
        return [c for c in self.quota if not self.is_full(c)]

    def take(self, cell: Cell) -> bool:
        """Consume one slot. False when the cell is full or unknown."""
        if self.room(cell) <= 0:
            return False
        self.filled[cell] = self.filled.get(cell, 0) + 1
        return True

    def overflow(self) -> dict[str, int]:
        """Cells filled beyond their quota. Must always be empty."""
        return {f"{d}/{lg}/K{k}": self.filled[(d, lg, k)] - self.quota[(d, lg, k)]
                for (d, lg, k) in self.quota
                if self.filled.get((d, lg, k), 0) > self.quota[(d, lg, k)]}

    def report(self) -> dict:
        return {
            "cells": len(self.quota),
            "total": self.total(),
            "taken": self.taken(),
            "full": self.full(),
            "unfilled_cells": len(self.open_cells()),
            "overflow": self.overflow(),
        }
