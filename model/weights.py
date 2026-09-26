"""Local backbone weight store (#T-torch-stack).

Real encoder weights live OUTSIDE git, under ``artifacts/weights/<id>/``
(see #T-repo-clean). This module is the single place that knows which
revision of which repo is pinned, downloads it, and verifies it by
SHA-256 before anything is allowed to load it.

External-dependency gate (STUB rules): the download needs network and,
for gated repos, an HF token. Both are gated on env vars —
``JEV_ALLOW_DOWNLOAD=0`` forbids any network call, ``HF_TOKEN`` /
``HUGGING_FACE_HUB_TOKEN`` supply the credential when a repo is gated.
When the gate is closed and the files are not already on disk, the fetch
returns a logged ``stub`` status instead of failing or, worse, inventing
a hash. No business logic is stubbed: verification always runs against
real bytes.

Two registries, one store: `BACKBONES` are the encoders a pointer head
sits on, `SCORERS` the cross-encoder NLI checkpoints of #T-ce-scorer.
`fetch`/`verify` treat them alike; only the walkers differ.

CLI:
    .venv-train/bin/python -m model.weights fetch
    .venv-train/bin/python -m model.weights verify
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_ROOT = os.path.join(ROOT, "artifacts", "weights")

# Pinned by commit sha, never by a moving branch: a live URL never
# reproduces a run (source-register rule).
BACKBONES = {
    "ettin-68m": {
        "repo": "jhu-clsp/ettin-encoder-68m",
        "revision": "ac19ae4bc51093b31c475665ac872a936d056cc2",
        "license": "MIT",
        "params_m": 68,
        "gated": False,
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "pytorch_model.bin"),
    },
    "modernbert-base": {
        "repo": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "license": "Apache-2.0",
        "params_m": 149,
        "gated": False,
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "model.safetensors"),
    },
}

# Cross-encoder scorer checkpoints (#T-ce-scorer). A SEPARATE registry on
# purpose: these are sequence-classification NLI heads, not encoders that
# `model.encoder.load_backbone` puts a pointer head on. `tools/smoke_torch`
# and the `model.bakeoff` Pareto walk BACKBONES and must not pick them up.
SCORERS = {
    "minilmv2-l6-mnli-xnli": {
        "repo": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        "revision": "0a71e92a985b6e1ad1828cf67ce9c459639c1dca",
        "license": "MIT",
        "params_m": 107,
        "gated": False,
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "sentencepiece.bpe.model",
                  "model.safetensors"),
    },
    "modernbert-zeroshot-v2": {
        "repo": "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        "revision": "d421c4545a438fd006fb43f8b981c5d908faa1e1",
        "license": "Apache-2.0",
        "params_m": 150,
        "gated": False,
        # Its published training mix INCLUDES BANKING77: this checkpoint
        # may NOT be presented as clean transfer to that benchmark
        # (plan-recuperacion-2026-09-24 §5).
        "contaminated_benchmarks": ("banking77",),
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "model.safetensors"),
    },
}

MANIFEST = "manifest.json"


def spec_for(weight_id: str) -> dict:
    """The pinned spec of a backbone OR a scorer — the registries are one
    store on disk (`artifacts/weights/<id>/`) and one fetch/verify path."""
    if weight_id in BACKBONES:
        return BACKBONES[weight_id]
    if weight_id in SCORERS:
        return SCORERS[weight_id]
    raise KeyError(f"unknown weight id {weight_id!r}; known: "
                   f"{sorted(list(BACKBONES) + list(SCORERS))}")


def weights_dir(backbone_id: str) -> str:
    return os.path.join(WEIGHTS_ROOT, backbone_id)


def manifest_path(backbone_id: str) -> str:
    return os.path.join(weights_dir(backbone_id), MANIFEST)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_allowed() -> bool:
    """Network gate. Closed by JEV_ALLOW_DOWNLOAD=0 or JEV_OFFLINE=1."""
    if os.environ.get("JEV_OFFLINE") == "1":
        return False
    return os.environ.get("JEV_ALLOW_DOWNLOAD", "1") != "0"


def hf_token() -> str | None:
    """Credential gate: only needed for gated repos; None is fine here."""
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)


def _log(msg: str) -> None:
    print(f"[weights] {msg}", flush=True)


def _present(backbone_id: str) -> bool:
    spec = spec_for(backbone_id)
    d = weights_dir(backbone_id)
    return all(os.path.exists(os.path.join(d, f)) for f in spec["files"])


def fetch(backbone_id: str) -> dict:
    """Materialise one backbone into artifacts/weights/<id>/.

    Returns a status dict: ``ok`` (bytes on disk, manifest written) or
    ``stub`` (gate closed and nothing cached — logged, never faked).
    """
    spec = spec_for(backbone_id)
    dest = weights_dir(backbone_id)
    os.makedirs(dest, exist_ok=True)

    if _present(backbone_id):
        _log(f"{backbone_id}: already on disk, re-hashing")
    elif not download_allowed():
        reason = "JEV_ALLOW_DOWNLOAD=0/JEV_OFFLINE=1 and no local copy"
        _log(f"{backbone_id}: STUB — {reason}")
        return {"id": backbone_id, "status": "stub", "reason": reason}
    elif spec["gated"] and not hf_token():
        reason = "gated repo and no HF_TOKEN in env"
        _log(f"{backbone_id}: STUB — {reason}")
        return {"id": backbone_id, "status": "stub", "reason": reason}
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # torch stack not installed
            reason = f"huggingface_hub missing ({exc})"
            _log(f"{backbone_id}: STUB — {reason}")
            return {"id": backbone_id, "status": "stub", "reason": reason}
        for fname in spec["files"]:
            _log(f"{backbone_id}: fetching {fname}")
            try:
                src = hf_hub_download(repo_id=spec["repo"], filename=fname,
                                      revision=spec["revision"],
                                      token=hf_token())
            except Exception as exc:  # network/auth/404 — never fake it
                reason = f"{fname}: {type(exc).__name__}: {exc}"
                _log(f"{backbone_id}: STUB — {reason}")
                return {"id": backbone_id, "status": "stub",
                        "reason": reason}
            shutil.copyfile(src, os.path.join(dest, fname))

    files = {f: {"sha256": sha256_file(os.path.join(dest, f)),
                 "bytes": os.path.getsize(os.path.join(dest, f))}
             for f in spec["files"]}
    manifest = {
        "id": backbone_id,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "license": spec["license"],
        "params_m": spec["params_m"],
        "fetched_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    with open(manifest_path(backbone_id), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    _log(f"{backbone_id}: ok, {len(files)} files, manifest written")
    return {"id": backbone_id, "status": "ok", "manifest": manifest}


def load_manifest(backbone_id: str) -> dict | None:
    path = manifest_path(backbone_id)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def verify(backbone_id: str) -> dict:
    """Re-hash every pinned file against the manifest.

    ``ok`` is False on ANY missing file or hash mismatch — callers must
    refuse to load the backbone in that case.
    """
    spec = spec_for(backbone_id)
    manifest = load_manifest(backbone_id)
    if manifest is None:
        return {"id": backbone_id, "ok": False, "reason": "no manifest",
                "files": {}}
    if manifest.get("revision") != spec["revision"]:
        return {"id": backbone_id, "ok": False,
                "reason": (f"manifest revision {manifest.get('revision')} "
                           f"!= pinned {spec['revision']}"),
                "files": {}}
    d = weights_dir(backbone_id)
    results, ok = {}, True
    for fname in spec["files"]:
        path = os.path.join(d, fname)
        want = manifest["files"].get(fname, {}).get("sha256")
        if not os.path.exists(path):
            results[fname] = {"state": "missing", "expected": want}
            ok = False
            continue
        got = sha256_file(path)
        good = bool(want) and got == want
        results[fname] = {"state": "ok" if good else "mismatch",
                          "expected": want, "actual": got}
        ok = ok and good
    return {"id": backbone_id, "ok": ok, "revision": spec["revision"],
            "license": spec["license"], "files": results,
            "reason": None if ok else "sha256 mismatch or missing file"}


def require_verified(backbone_id: str) -> str:
    """Return the weights dir, or raise if the hashes don't check out."""
    report = verify(backbone_id)
    if not report["ok"]:
        raise RuntimeError(
            f"{backbone_id}: refusing to load — {report['reason']}: "
            f"{json.dumps(report['files'], sort_keys=True)}")
    return weights_dir(backbone_id)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "fetch"
    ids = argv[2:] or list(BACKBONES) + list(SCORERS)
    if cmd == "fetch":
        results = [fetch(i) for i in ids]
        stubs = [r for r in results if r["status"] == "stub"]
        return 1 if stubs else 0
    if cmd == "verify":
        reports = [verify(i) for i in ids]
        for r in reports:
            _log(f"{r['id']}: {'PASS' if r['ok'] else 'FAIL'} "
                 f"{r['reason'] or ''}".rstrip())
        return 0 if all(r["ok"] for r in reports) else 1
    _log(f"unknown command {cmd!r}; use fetch|verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
