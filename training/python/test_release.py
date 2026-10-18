"""Research release and observability verification (#T-release).

Builds the atomic release bundle (``artifacts/gates/T-release/release.json``)
and proves, with measured values only:

- every evidence file the bundle cites exists and its sha256 matches
  (weights, licenses, dataset manifests, hashes, calibration report);
- every numeric claim in the bundle (temperature, kappa, top-2, cold
  verdict) equals the raw artifact it cites — no claim without evidence;
- the model card's claims resolve to files on disk with matching values;
- an offline install (fresh directory, hash re-verified) re-executes the
  golden decisions identically;
- licenses: SBOM pins every dependency, workspace crates declare SPDX,
  dataset/model licenses are disclosed in the model card — and no secret
  or state content leaks into the bundle or the Prometheus exposition;
- ``GET /metrics`` on a live server emits counters only;
- the Pod train flow destroys the Pod even when the run fails (stub;
  live runs stay gated behind ``RUNPOD_LIVE=1`` + operator authorization).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
from training.python.parity.reference import MATRIX, case
from training.python.tools.pod_stub import FakePodClient, live_enabled, run_ephemeral

OUT_DIR = os.path.join(REPO, "artifacts", "gates", "T-release")
RELEASE = os.path.join(OUT_DIR, "release.json")
CARD = os.path.join(REPO, ".meshkore", "docs", "model-card.md")
METRICS_PORT = 18082
SENTINEL = 6.62607015

# (claim, repo-relative evidence path). Every claim must resolve.
EVIDENCE = [
    ("calibration", "artifacts/gates/T-calib/calibration.json"),
    ("calibration-report", "artifacts/gates/T-calib/report.json"),
    ("gold-pilot", "artifacts/gates/T-gold/report.json"),
    ("bakeoff", "artifacts/gates/T-bakeoff/report.json"),
    ("cold-start", "artifacts/gates/T-cloud-api/coldstart.json"),
    ("sbom", "sbom.json"),
    ("model-card", ".meshkore/docs/model-card.md"),
    ("pod-train", ".meshkore/docs/pod-train.md"),
    ("api-v1", ".meshkore/docs/api-v1.md"),
    ("sources", ".meshkore/docs/source-register.md"),
    ("dataset-registry", "data/registry.py"),
]

SECRET_PATTERNS = [
    r"(?i)api[_-]?key\s*[:=]",
    r"(?i)bearer\s+[A-Za-z0-9._~-]{8,}",
    r"(?i)password\s*[:=]",
    r"sk-[A-Za-z0-9]{8,}",
    r"BEGIN [A-Z ]*PRIVATE KEY",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
        capture_output=True, text=True, timeout=30,
    )
    return out.stdout.strip() if out.returncode == 0 else "uncommitted"


def build_release() -> dict:
    def load(rel: str):
        with open(os.path.join(REPO, rel)) as f:
            text = f.read()
        try:
            return json.loads(text)
        except ValueError:
            return text
    raw = {rel: load(rel) for _, rel in EVIDENCE}
    calib = raw["artifacts/gates/T-calib/calibration.json"]
    gold = raw["artifacts/gates/T-gold/report.json"]
    bakeoff = raw["artifacts/gates/T-bakeoff/report.json"]
    cold = raw["artifacts/gates/T-cloud-api/coldstart.json"]
    bundle = {
        "format": 1,
        "task": "T-release",
        "commit": git_commit(),
        "predictor": calib["predictor"],
        "licenses": {
            "workspace": "MIT OR Apache-2.0",
            "evidence": ".meshkore/docs/model-card.md",
        },
        "numbers": {
            "temperature": calib["global"]["temperature"],
            "nll_before": calib["global"]["nll_before"],
            "nll_after": calib["global"]["nll_after"],
            "cohen_kappa": gold["iaa"]["cohen_kappa"],
            "exact_agree_rate": gold["iaa"]["exact_agree_rate"],
            "top2": bakeoff["top2"],
            "cold_verdict": cold["verdict"],
            "cold_budget_s": cold["budget_s"],
        },
        "evidence": [
            {"claim": claim, "path": rel,
             "sha256": sha256_file(os.path.join(REPO, rel))}
            for claim, rel in EVIDENCE
        ],
        "golden": [
            {"name": name, "seed": s, "dim": d, "n": n,
             "probs": case(s, d, n, False)}
            for (name, s, d, n, _k) in MATRIX[:3]
        ],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RELEASE, "w") as f:
        json.dump(bundle, f, indent=2)
        f.write("\n")
    return bundle


def fetch_text(base: str, path: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read().decode()


def post(base: str, path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def wait_ready(base: str, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/ready", timeout=10) as r:
                if r.status == 200 and json.loads(r.read().decode()).get("ready"):
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"server at {base} never ready")


class TestRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_release()

    def test_evidence_hashes_match_and_numbers_equal_raw(self):
        for item in self.bundle["evidence"]:
            p = os.path.join(REPO, item["path"])
            self.assertTrue(os.path.isfile(p), item["path"])
            self.assertEqual(sha256_file(p), item["sha256"], item["claim"])
        n = self.bundle["numbers"]
        calib = json.load(open(
            os.path.join(REPO, "artifacts/gates/T-calib/calibration.json")))
        gold = json.load(open(
            os.path.join(REPO, "artifacts/gates/T-gold/report.json")))
        bakeoff = json.load(open(
            os.path.join(REPO, "artifacts/gates/T-bakeoff/report.json")))
        cold = json.load(open(os.path.join(
            REPO, "artifacts/gates/T-cloud-api/coldstart.json")))
        self.assertEqual(n["temperature"], calib["global"]["temperature"])
        self.assertEqual(n["cohen_kappa"], gold["iaa"]["cohen_kappa"])
        self.assertEqual(n["exact_agree_rate"], gold["iaa"]["exact_agree_rate"])
        self.assertEqual(n["top2"], bakeoff["top2"])
        self.assertEqual(n["cold_verdict"], cold["verdict"])
        self.assertEqual(n["cold_verdict"], "GO")

    def test_model_card_claims_resolve_to_evidence(self):
        self.assertTrue(os.path.isfile(CARD))
        card = open(CARD).read()
        n = self.bundle["numbers"]
        for value in (str(n["temperature"]), str(n["cohen_kappa"]),
                      str(n["exact_agree_rate"]), n["cold_verdict"],
                      *n["top2"]):
            self.assertIn(value, card, f"card must state {value!r}")
        for m in re.findall(r"(artifacts/[^\s)`\"]+|\.meshkore/docs/[^\s)`\"]+)",
                            card):
            target = os.path.join(REPO, m.rstrip("/"))
            self.assertTrue(os.path.isfile(target) or os.path.isdir(target),
                            f"card cites missing path {m}")

    def test_offline_install_golden_licenses_no_secrets(self):
        tmp = tempfile.mkdtemp(prefix="jev-release-")
        self.addCleanup(shutil.rmtree, tmp, True)
        for item in self.bundle["evidence"]:
            src = os.path.join(REPO, item["path"])
            dst = os.path.join(tmp, item["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            self.assertEqual(sha256_file(dst), item["sha256"])  # re-verified
        ref_src = os.path.join(REPO, "training", "python", "parity",
                               "reference.py")
        ref_dst = os.path.join(tmp, "reference.py")
        shutil.copy2(ref_src, ref_dst)
        spec = importlib.util.spec_from_file_location("offline_ref", ref_dst)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for g in self.bundle["golden"]:
            self.assertEqual(mod.case(g["seed"], g["dim"], g["n"], False),
                             g["probs"], g["name"])
        sbom = json.load(open(os.path.join(tmp, "sbom.json")))
        pkgs = sbom.get("packages", [])
        self.assertTrue(pkgs, "SBOM must list packages")
        for p in pkgs:  # every dependency pinned: reproducibility is the claim
            self.assertTrue(p.get("name") and p.get("version"),
                            f"unpinned SBOM entry: {p}")
        spdx = set()
        for manifest in ("Cargo.toml",):
            for line in open(os.path.join(REPO, manifest)):
                if line.startswith("license"):
                    spdx.add(line.strip())
        import glob as _glob
        for crate in _glob.glob(os.path.join(REPO, "crates", "*", "Cargo.toml")):
            for line in open(crate):
                if line.startswith("license"):
                    spdx.add(line.strip())
        self.assertTrue(spdx, "workspace crates must declare a license")
        self.assertTrue(all("MIT OR Apache-2.0" in s for s in spdx), spdx)
        card = open(CARD).read()  # dataset/model licenses live in the card
        for lic in ("CC-BY-4.0", "CC-BY-SA-3.0", "CC0", "MIT", "Apache-2.0"):
            self.assertIn(lic, card, f"card must disclose {lic}")
        blob = open(RELEASE).read() + open(CARD).read()
        for pat in SECRET_PATTERNS:
            self.assertIsNone(re.search(pat, blob), f"secret pattern {pat}")
        self.assertNotIn("state-content", blob)

    def test_metrics_live_counters_without_user_content(self):
        subprocess.run(["cargo", "build", "-p", "jev-cli"], cwd=REPO,
                       check=True, capture_output=True)
        proc = subprocess.Popen(
            [os.path.join(REPO, "target", "debug", "jevclone"),
             "serve", "--listen", f"127.0.0.1:{METRICS_PORT}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.terminate)
        base = f"http://127.0.0.1:{METRICS_PORT}"
        wait_ready(base)
        q = {"model_version": "m1",
             "options": [{"id": "yes"}, {"id": "no"}],
             "weights": [SENTINEL] * 8, "bias": [0.0, 0.0],
             "input": [SENTINEL] * 4}
        status, _ = post(base, "/v1/choice", q)
        self.assertEqual(status, 200)
        status, ctype, body = fetch_text(base, "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", ctype)
        for metric in ("jev_requests_total", "jev_errors_total",
                       "jev_cache_hits_total", "jev_cache_misses_total",
                       "jev_infer_us_sum"):
            self.assertIn(metric, body)
        m = re.search(r"jev_requests_total (\d+)", body)
        self.assertIsNotNone(m)
        self.assertGreaterEqual(int(m.group(1)), 1)
        self.assertNotIn(str(SENTINEL), body)  # no state echo in telemetry

    def test_pod_stub_always_destroys_and_live_gated(self):
        client = FakePodClient()
        out = run_ephemeral(client, "img:train", "run --curriculum")
        self.assertTrue(out.startswith("ok:"))
        self.assertEqual(client.destroyed, client.created)
        failing = FakePodClient()
        with self.assertRaises(RuntimeError):
            run_ephemeral(failing, "img:train", "fail")
        self.assertEqual(failing.destroyed, failing.created)
        self.assertFalse(live_enabled(), "gates must take the stub path")


if __name__ == "__main__":
    unittest.main()
