"""Tests for #T-torch-stack (stdlib unittest).

Two layers:

* hash/gate logic — runs anywhere, no torch, no weights: it fabricates a
  weights dir in a tmpdir and checks that a tampered byte is refused;
* the real stack — skipped when torch or the downloaded weights are
  absent, so a clean checkout still runs the suite, but NEVER faked.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from . import weights

try:  # the real stack is optional at test time, never stubbed
    import torch
    from .encoder import (encode_state, first_row, load_backbone,
                          prefetch_path, state_text)
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - env without the training venv
    HAVE_TORCH = False

REAL_WEIGHTS = [b for b in weights.BACKBONES
                if os.path.exists(weights.manifest_path(b))]


def _fake_store(root: str, backbone_id: str = "ettin-68m") -> str:
    """Build a minimal, self-consistent weights dir under `root`."""
    d = os.path.join(root, backbone_id)
    os.makedirs(d, exist_ok=True)
    spec = weights.BACKBONES[backbone_id]
    files = {}
    for i, fname in enumerate(spec["files"]):
        path = os.path.join(d, fname)
        with open(path, "wb") as fh:
            fh.write(f"payload-{i}".encode())
        files[fname] = {"sha256": weights.sha256_file(path),
                        "bytes": os.path.getsize(path)}
    with open(os.path.join(d, weights.MANIFEST), "w") as fh:
        json.dump({"id": backbone_id, "repo": spec["repo"],
                   "revision": spec["revision"], "license": spec["license"],
                   "params_m": spec["params_m"],
                   "fetched_utc": "2026-09-21T00:00:00Z",
                   "files": files}, fh)
    return d


class TestRegistry(unittest.TestCase):
    def test_backbones_pinned_by_commit(self):
        self.assertIn("ettin-68m", weights.BACKBONES)
        self.assertIn("modernbert-base", weights.BACKBONES)
        for bid, spec in weights.BACKBONES.items():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$",
                             f"{bid} must pin a commit sha, not a branch")
            self.assertIn("tokenizer.json", spec["files"])
            self.assertIn("config.json", spec["files"])
            self.assertTrue(any(f.endswith((".bin", ".safetensors"))
                                for f in spec["files"]))

    def test_download_gate_env(self):
        with mock.patch.dict(os.environ, {"JEV_ALLOW_DOWNLOAD": "0"}):
            self.assertFalse(weights.download_allowed())
        with mock.patch.dict(os.environ, {"JEV_OFFLINE": "1",
                                          "JEV_ALLOW_DOWNLOAD": "1"}):
            self.assertFalse(weights.download_allowed())
        with mock.patch.dict(os.environ, {"JEV_ALLOW_DOWNLOAD": "1",
                                          "JEV_OFFLINE": "0"}):
            self.assertTrue(weights.download_allowed())


class TestHashVerification(unittest.TestCase):
    def test_clean_store_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            _fake_store(root)
            with mock.patch.object(weights, "WEIGHTS_ROOT", root):
                report = weights.verify("ettin-68m")
        self.assertTrue(report["ok"], report)
        self.assertTrue(all(f["state"] == "ok"
                            for f in report["files"].values()))

    def test_tampered_byte_fails_and_blocks_load(self):
        with tempfile.TemporaryDirectory() as root:
            d = _fake_store(root)
            target = os.path.join(d, "tokenizer.json")
            with open(target, "ab") as fh:
                fh.write(b"x")
            with mock.patch.object(weights, "WEIGHTS_ROOT", root):
                report = weights.verify("ettin-68m")
                with self.assertRaises(RuntimeError):
                    weights.require_verified("ettin-68m")
        self.assertFalse(report["ok"])
        self.assertEqual(report["files"]["tokenizer.json"]["state"],
                         "mismatch")

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as root:
            d = _fake_store(root)
            os.remove(os.path.join(d, "config.json"))
            with mock.patch.object(weights, "WEIGHTS_ROOT", root):
                report = weights.verify("ettin-68m")
        self.assertFalse(report["ok"])
        self.assertEqual(report["files"]["config.json"]["state"], "missing")

    def test_manifest_revision_must_match_pin(self):
        with tempfile.TemporaryDirectory() as root:
            d = _fake_store(root)
            mpath = os.path.join(d, weights.MANIFEST)
            with open(mpath) as fh:
                manifest = json.load(fh)
            manifest["revision"] = "0" * 40
            with open(mpath, "w") as fh:
                json.dump(manifest, fh)
            with mock.patch.object(weights, "WEIGHTS_ROOT", root):
                report = weights.verify("ettin-68m")
        self.assertFalse(report["ok"])
        self.assertIn("revision", report["reason"])

    def test_no_manifest_is_not_ok(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(weights, "WEIGHTS_ROOT", root):
                report = weights.verify("ettin-68m")
        self.assertFalse(report["ok"])


class TestStubGate(unittest.TestCase):
    def test_closed_gate_without_local_copy_stubs(self):
        """No network + nothing cached => logged stub, never a fake hash."""
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(weights, "WEIGHTS_ROOT", root), \
                    mock.patch.dict(os.environ,
                                    {"JEV_ALLOW_DOWNLOAD": "0"}):
                result = weights.fetch("ettin-68m")
                self.assertFalse(
                    os.path.exists(weights.manifest_path("ettin-68m")))
        self.assertEqual(result["status"], "stub")
        self.assertIn("reason", result)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (.venv-train)")
class TestTorchAvailable(unittest.TestCase):
    def test_torch_and_transformers_import(self):
        import transformers
        self.assertTrue(torch.__version__)
        self.assertTrue(transformers.__version__)

    def test_accelerator_reported_honestly(self):
        from .encoder import device_report, pick_device
        rep = device_report()
        self.assertEqual(rep["mps_available"],
                         torch.backends.mps.is_available())
        expected = "mps" if torch.backends.mps.is_available() else "cpu"
        self.assertEqual(str(pick_device()), expected)


@unittest.skipUnless(HAVE_TORCH and REAL_WEIGHTS,
                     "downloaded weights not present "
                     "(python -m model.weights fetch)")
class TestRealForwardPass(unittest.TestCase):
    def test_every_downloaded_backbone_matches_its_manifest(self):
        for bid in REAL_WEIGHTS:
            with self.subTest(bid):
                report = weights.verify(bid)
                self.assertTrue(report["ok"], report)

    def test_encode_state_shape_on_cpu(self):
        row = first_row(prefetch_path("banking77"))
        backbone = load_backbone(REAL_WEIGHTS[0], "cpu")
        out = encode_state(backbone, state_text(row))
        self.assertEqual(out["tokens"].shape[0], 1)
        self.assertEqual(out["tokens"].shape[1], out["n_tokens"])
        self.assertEqual(out["tokens"].shape[2], backbone.hidden_size)
        self.assertEqual(tuple(out["pooled"].shape),
                         (1, backbone.hidden_size))
        self.assertTrue(bool(out["pooled"].isfinite().all()))
        self.assertGreater(out["ms"], 0.0)

    def test_encode_state_rejects_empty(self):
        backbone = load_backbone(REAL_WEIGHTS[0], "cpu")
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                encode_state(backbone, bad)


if __name__ == "__main__":
    unittest.main()
