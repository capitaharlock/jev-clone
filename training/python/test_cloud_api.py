"""Contract tests for the public decision API V1 (#T-cloud-api).

Same golden corpus runs against the local binary and the Docker image;
both must answer with identical semantics. Also covers schema/versions,
order + IDs, batch multi-Q, `unknown`, errors, limits, timeouts and the
no-state-in-logs rule (error bodies must not echo weights/inputs), plus
image inspection (no Python, SBOM present) and a measured cold-start
written to ``artifacts/gates/T-cloud-api/coldstart.json``. No number is
invented: every field is measured on this machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import unittest
import urllib.request

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT_DIR = os.path.join(ROOT, "artifacts", "gates", "T-cloud-api")
COLDSTART = os.path.join(OUT_DIR, "coldstart.json")
IMAGE = "jevclone:gate"
LOCAL_PORT = 18080
DOCKER_PORT = 18081
COLD_BUDGET_S = 30.0
SENTINEL = 7.777

CORPUS = [
    {
        "model_version": "m1",
        "options": [{"id": "yes"}, {"id": "no"}],
        "weights": [0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
        "bias": [0.1, -0.1],
        "input": [1.0, 0.5, -0.5, 0.25],
    },
    {
        "model_version": "m1",
        "options": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "weights": [0.1 * i - 0.3 for i in range(12)],
        "bias": [0.0, 0.05, -0.05],
        "input": [0.2, -0.4, 0.6, -0.1],
        "threshold": 0.0,
    },
    {
        "model_version": "m2",
        "options": [{"id": "x"}, {"id": "y"}],
        "weights": [0.3, 0.1, -0.2, 0.4],
        "bias": [0.0, 0.0],
        "input": [1.0, -1.0],
        "threshold": 1.0,  # forces unknown
    },
]


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


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        raw = r.read().decode()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def wait_ready(base: str, timeout: float = 60.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            status, body = get(base, "/ready")
            if status == 200 and isinstance(body, dict) and body.get("ready"):
                return time.time() - t0
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"server at {base} never ready")


class TestCloudApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["cargo", "build", "-p", "jev-cli"], cwd=ROOT, check=True,
            capture_output=True,
        )
        cls.local = subprocess.Popen(
            [os.path.join(ROOT, "target", "debug", "jevclone"),
             "serve", "--listen", f"127.0.0.1:{LOCAL_PORT}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cls.addClassCleanup(cls.local.terminate)
        cls.base = f"http://127.0.0.1:{LOCAL_PORT}"
        wait_ready(cls.base)

    def test_health_ready_schema(self):
        status, body = get(self.base, "/health")
        self.assertEqual((status, body), (200, "ok"))
        status, body = get(self.base, "/ready")
        self.assertEqual(status, 200)
        self.assertTrue(body["ready"])
        self.assertEqual(body["schema"], "v1")

    def test_corpus_choice_contract(self):
        for q in CORPUS:
            status, body = post(self.base, "/v1/choice", q)
            self.assertEqual(status, 200, body)
            self.assertEqual(body["schema"], "v1")
            self.assertEqual(body["model_version"], q["model_version"])
            self.assertEqual(
                [d["id"] for d in body["distribution"]],
                [o["id"] for o in q["options"]],
            )
            self.assertAlmostEqual(
                sum(d["prob"] for d in body["distribution"]), 1.0, places=5
            )
            self.assertEqual(body["unknown"], body["choice"] is None)
        # threshold 1.0 forces unknown
        _, forced = post(self.base, "/v1/choice", CORPUS[2])
        self.assertTrue(forced["unknown"])
        self.assertIsNone(forced["choice"])

    def test_batch_multi_q_order_and_partial_error(self):
        bad = dict(CORPUS[0])
        bad["options"] = []
        status, body = post(
            self.base, "/v1/batch", {"queries": [CORPUS[0], CORPUS[1], bad]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["results"]), 3)
        self.assertEqual(body["results"][0]["status"], "ok")
        self.assertEqual(body["results"][1]["status"], "ok")
        self.assertEqual(body["results"][2]["status"], "err")
        first = post(self.base, "/v1/choice", CORPUS[0])[1]
        self.assertEqual(body["results"][0]["distribution"], first["distribution"])

    def test_errors_versions_limits_no_state_echo(self):
        bad_schema = dict(CORPUS[0], schema="v9")
        status, body = post(self.base, "/v1/choice", bad_schema)
        self.assertEqual(status, 422)
        self.assertEqual(body["code"], "bad-schema")
        dup = dict(CORPUS[0])
        dup["options"] = [{"id": "a"}, {"id": "a"}]
        status, body = post(self.base, "/v1/choice", dup)
        self.assertEqual(status, 422)
        raw = json.dumps(body)
        self.assertNotIn("0.5", raw)  # no weight echo
        sentinel = dict(CORPUS[0])
        sentinel["weights"] = [SENTINEL] * 8
        sentinel["options"] = []
        _, body = post(self.base, "/v1/choice", sentinel)
        self.assertNotIn(str(SENTINEL), json.dumps(body))
        big = {"queries": [CORPUS[0]] * 65}
        status, _ = post(self.base, "/v1/batch", big)
        self.assertEqual(status, 422)

    def test_runpod_stub_envelope(self):
        status, body = post(self.base, "/runpod", {"input": CORPUS[0], "id": "job-1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], "job-1")
        self.assertEqual(body["output"]["schema"], "v1")
        self.assertEqual(len(body["output"]["distribution"]), 2)
        status, _ = post(self.base, "/runpod", {"id": "x"})
        self.assertEqual(status, 400)

    def test_docker_parity_image_coldstart(self):
        has_image = subprocess.run(
            ["docker", "images", "-q", IMAGE], capture_output=True, text=True
        ).stdout.strip()
        if not has_image:
            subprocess.run(
                ["docker", "build", "-t", IMAGE, "."], cwd=ROOT, check=True,
                capture_output=True,
            )
        # image inspection: no python, SBOM present, matches repo
        py = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "sh", IMAGE,
             "-c", "command -v python3 || echo none"],
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(py, "none")
        sbom = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, "/sbom.json"],
            capture_output=True, text=True,
        ).stdout
        repo_sbom = open(os.path.join(ROOT, "sbom.json")).read()
        self.assertEqual(json.loads(sbom), json.loads(repo_sbom))
        # cold-start: spawn -> /ready, measured
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "-p", f"{DOCKER_PORT}:8080", IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.terminate)
        t0 = time.time()
        elapsed = wait_ready(f"http://127.0.0.1:{DOCKER_PORT}", timeout=120.0)
        wall = time.time() - t0
        # semantic parity on the golden corpus
        dbase = f"http://127.0.0.1:{DOCKER_PORT}"
        for q in CORPUS:
            _, local = post(self.base, "/v1/choice", q)
            status, remote = post(dbase, "/v1/choice", q)
            self.assertEqual(status, 200)
            self.assertEqual(remote["distribution"], local["distribution"])
            self.assertEqual(remote["choice"], local["choice"])
            self.assertEqual(remote["unknown"], local["unknown"])
        verdict = "GO" if elapsed <= COLD_BUDGET_S else "NO-GO"
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump(
            {"cold_start_s": elapsed, "wall_s": wall,
             "budget_s": COLD_BUDGET_S, "verdict": verdict,
             "image": IMAGE},
            open(COLDSTART, "w"), indent=2,
        )
        self.assertEqual(verdict, "GO", f"cold-start {elapsed:.1f}s over budget")


if __name__ == "__main__":
    unittest.main()
