"""Tiny-model reference for the Rust↔Python parity harness (#T-rust-skel).

Implements the SAME math as `crates/jev-model/src/lib.rs` using only the
standard library (no torch needed for the skeleton gate): LCG weights,
logits = W·x + b, softmax; FP16 path rounds every input through binary16
via `struct '<e'`. The Rust integration test runs this script with
`--matrix` and compares its JSON output against the Rust computation.

Production note: this reference stands in for PyTorch until the first
real backbone lands (#T-bakeoff); the Rust side stands in for Candle.
The harness shape (matrix × dtypes × tolerances) does not change then.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys

M32, A32 = 1103515245, 12345


def lcg(seed: int, n: int) -> list[float]:
    s = seed
    out = []
    for _ in range(n):
        s = (A32 + M32 * s) & 0x7FFFFFFF
        out.append(s / 2147483648.0 * 2.0 - 1.0)
    return out


def to_f16(x: float) -> float:
    return struct.unpack("<e", struct.pack("<e", x))[0]


def logits(w: list[float], b: list[float], x: list[float], n_options: int) -> list[float]:
    dim = len(x)
    return [
        sum(w[i * dim + j] * x[j] for j in range(dim)) + b[i]
        for i in range(n_options)
    ]


def softmax(ls: list[float]) -> list[float]:
    m = max(ls)
    exps = [math.exp(v - m) for v in ls]
    s = sum(exps)
    return [e / s for e in exps]


def case(seed: int, dim: int, n_options: int, use_f16: bool) -> list[float]:
    vals = lcg(seed, n_options * dim + n_options + dim)
    w, b = vals[: n_options * dim], vals[n_options * dim: n_options * dim + n_options]
    x = vals[n_options * dim + n_options:]
    if use_f16:
        w = [to_f16(v) for v in w]
        b = [to_f16(v) for v in b]
        x = [to_f16(v) for v in x]
    # NOTE: CPython floats are f64; the f32/f16 paths they emulate are
    # checked on the Rust side within TOL_F32/TOL_F16, not bit-exact here.
    return softmax(logits(w, b, x, n_options))


MATRIX = [
    # (name, seed, dim, n_options, kind)
    ("short-2-choice", 11, 4, 2, "choice"),
    ("short-2b-boolean", 12, 4, 2, "boolean"),
    ("short-8-choice", 13, 4, 8, "choice"),
    ("long-2-boolean", 14, 16, 2, "boolean"),
    ("long-8-choice", 15, 16, 8, "choice"),
    ("long-32-choice", 16, 16, 32, "choice"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--case", default="")
    ap.add_argument("--f16", action="store_true")
    args = ap.parse_args()
    if args.matrix:
        print(json.dumps([
            {"name": n, "dim": d, "options": o, "kind": k,
             "fp32": case(s, d, o, False), "fp16": case(s, d, o, True)}
            for (n, s, d, o, k) in MATRIX
        ]))
        return 0
    for (n, s, d, o, _k) in MATRIX:
        if n == args.case:
            print(json.dumps(case(s, d, o, args.f16)))
            return 0
    print(f"unknown case: {args.case}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
