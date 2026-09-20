"""Quantization + ONNX export backends, Python mirror of jev-quant (#T-quant-onnx).

Strict order: FP32 canonical -> FP16/BF16 reference -> INT8 dynamic ->
INT8 static (recalibrated) -> INT4 experimental. ONNX is a derived
manifest referencing the canonical hash.
"""

from __future__ import annotations

import hashlib
import json
import struct

try:
    import numpy as _np  # type: ignore

    _HAS_NP = True
except Exception:  # pragma: no cover
    _HAS_NP = False


def canonical_hash(weights: list) -> str:
    h = hashlib.sha256()
    for w in weights:
        h.update(struct.pack("<f", float(w)))
    return h.hexdigest()


def fp16_roundtrip(weights: list) -> list:
    if _HAS_NP:
        a = _np.asarray(weights, dtype=_np.float32).astype(_np.float16).astype(_np.float32)
        return [float(v) for v in a.tolist()]
    out = []
    for w in weights:
        out.append(struct.unpack("<f", struct.pack("<f", float(w)))[0])
    return out


def bf16_roundtrip(weights: list) -> list:
    out = []
    for w in weights:
        bits = struct.unpack("<I", struct.pack("<f", float(w)))[0]
        if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0:  # NaN: keep quiet
            bits = (bits & 0xFFFF0000) | 0x00400000
        else:
            bits &= 0xFFFF0000
        out.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return out


def int8_quantize_dynamic(weights: list) -> dict:
    amax = max((abs(float(w)) for w in weights), default=0.0)
    scale = 1.0 if amax == 0.0 else amax / 127.0
    data = [max(-128, min(127, round(float(w) / scale))) for w in weights]
    return {"scales": [scale], "data": data, "per_channel": False}


def int8_dequantize(q: dict, channels: int = 1) -> list:
    if q.get("per_channel"):
        cols = len(q["data"]) // channels
        return [v * q["scales"][i // cols] for i, v in enumerate(q["data"])]
    return [v * q["scales"][0] for v in q["data"]]


def int8_quantize_per_channel(weights: list, channels: int) -> dict:
    assert channels > 0 and len(weights) % channels == 0
    cols = len(weights) // channels
    scales, data = [], []
    for c in range(channels):
        row = [float(v) for v in weights[c * cols : (c + 1) * cols]]
        amax = max((abs(v) for v in row), default=0.0)
        scale = 1.0 if amax == 0.0 else amax / 127.0
        scales.append(scale)
        data += [max(-128, min(127, round(v / scale))) for v in row]
    return {"scales": scales, "data": data, "per_channel": True}


def int4_quantize_experimental(weights: list) -> dict:
    amax = max((abs(float(w)) for w in weights), default=0.0)
    scale = 1.0 if amax == 0.0 else amax / 7.0
    nibbles = [max(-8, min(7, round(float(w) / scale))) & 0x0F for w in weights]
    packed = []
    for i in range(0, len(nibbles), 2):
        hi = nibbles[i] << 4
        lo = nibbles[i + 1] if i + 1 < len(nibbles) else 0
        packed.append(hi | lo)
    return {"scale": scale, "packed": packed, "len": len(weights), "experimental": True}


def int4_dequantize(q: dict) -> list:
    out = []
    for b in q["packed"]:
        for nib in ((b >> 4) & 0x0F, b & 0x0F):
            if len(out) == q["len"]:
                break
            signed = nib - 16 if nib >= 8 else nib
            out.append(signed * q["scale"])
    return out


def error_stats(reference: list, candidate: list) -> tuple:
    assert len(reference) == len(candidate)
    errs = [abs(float(a) - float(b)) for a, b in zip(reference, candidate)]
    return max(errs), sum(errs) / len(errs)


def onnx_export_manifest(weights: list, buckets_kq: list, eps: list) -> dict:
    if not buckets_kq:
        raise ValueError("at least one K/Q bucket required")
    if any(b <= 0 or b > 131072 for b in buckets_kq):
        raise ValueError("invalid K/Q bucket: must be 1..=131072")
    m = {
        "opset": 17,
        "dynamic_axes": {"seq_k": list(buckets_kq), "seq_q": list(buckets_kq)},
        "buckets_kq": list(buckets_kq),
        "canonical_hash": canonical_hash(weights),
        "ep_available": list(eps),
    }
    back = json.loads(json.dumps(m))  # round-trip must be lossless
    assert back == m
    m["roundtrip_ok"] = True
    return m
