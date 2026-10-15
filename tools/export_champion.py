"""Export the promoted champion as a standalone TinyJev snapshot (#T-champion-export).

The champion is a rank-16 LoRA plus a retrained pointer head on top of the
TinyJev-4B v1 snapshot. Fusing the LoRA into the backbone gives a directory
with the same layout as the v1 snapshot (`model.safetensors`,
`head.safetensors`, config, tokenizer, `tinyjev.json`), so any loader of v1,
the Candle runtime included, reads the champion without knowing about LoRA.

The fuse runs on the CPU so it never competes with the autoloop for Metal.
A random-token batch is scored before and after the fuse; the export is
refused when the pointer logits move more than the fp16 tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.python import pointer_finetune as P  # noqa: E402

CHAMPION = ROOT / "artifacts" / "autoloop" / "champion"
OUT_ROOT = ROOT / "artifacts" / "runtime"
TOLERANCE = 5e-2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_logits(model, config: dict, seed: int = 7):
    import mlx.core as mx
    mx.random.seed(seed)
    tokens = mx.random.randint(0, config["vocab_size"], (2, 128))
    decide = mx.array([100, 120])
    opts = mx.array([[10 * i + 5 for i in range(P.OPTION_PAD)]] * 2)
    valid = mx.ones((2, P.OPTION_PAD))
    logits = P.pointer_logits(model, tokens, decide, opts, valid)
    mx.eval(logits)
    return logits


def fuse(model) -> int:
    from mlx.utils import tree_unflatten
    fused = [(name, module.fuse()) for name, module in model.backbone.named_modules()
             if hasattr(module, "fuse")]
    model.backbone.update_modules(tree_unflatten(fused))
    return len(fused)


def export(champion: Path, out_root: Path) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.set_default_device(mx.cpu)
    meta = json.loads((champion / "champion.json").read_text(encoding="utf-8"))
    adapter = champion / "adapters.safetensors"
    adapter_sha = sha256(adapter)
    out = out_root / f"champion-cycle-{int(meta['cycle']):04d}"
    manifest_path = out / "export.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("adapter_sha256") == adapter_sha:
            return {**previous, "status": "current"}

    sys.path.insert(0, str(P.TINYJEV_SRC))
    weights = P.resolve_weights()
    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    model = P.build_student(weights)
    P.load_adapter(model, adapter)
    before = probe_logits(model, config)
    layers = fuse(model)
    after = probe_logits(model, config)
    drift = float(mx.max(mx.abs(before - after)).item())
    if drift > TOLERANCE:
        raise RuntimeError(f"fused logits drift {drift:.4g} > {TOLERANCE}")

    backbone = dict(tree_flatten(model.backbone.parameters()))
    original = set(mx.load(str(weights / "model.safetensors")).keys())
    if set(backbone) != original:
        missing = sorted(original - set(backbone))[:4]
        extra = sorted(set(backbone) - original)[:4]
        raise RuntimeError(f"key mismatch: missing {missing}, extra {extra}")

    staging = out.with_name(out.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    mx.save_safetensors(str(staging / "model.safetensors"), backbone)
    mx.save_safetensors(str(staging / "head.safetensors"), {
        "q.weight": model.head_q.weight, "q.bias": model.head_q.bias,
        "k.weight": model.head_k.weight, "k.bias": model.head_k.bias,
    })
    for item in weights.iterdir():
        if item.suffix != ".safetensors" and item.name != ".gitattributes":
            shutil.copyfile(item.resolve(), staging / item.name)
    manifest = {
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "champion_cycle": meta["cycle"],
        "champion_source": meta.get("source"),
        "promoted_utc": meta.get("promoted_utc"),
        "base_snapshot": str(weights),
        "adapter_sha256": adapter_sha,
        "lora_layers_fused": layers,
        "probe_logit_drift": drift,
        "files": {item.name: sha256(item) for item in sorted(staging.iterdir())},
    }
    (staging / "export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    latest = out_root / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)
    return {**manifest, "status": "exported", "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--champion", type=Path, default=CHAMPION)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(export(args.champion, args.out), indent=2))


if __name__ == "__main__":
    main()
