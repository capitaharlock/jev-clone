---
title: Train-on-Pod flow (RunPod Pods, stub-tested)
updated: 2026-09-20
owner: general-09192230
---

# Train-on-Pod flow (RunPod Pods)

Cloud training on RunPod Pods is a fallback for when local iron is not
enough — it has not been needed for V1. The flow below is tested against
the stub in `training/python/tools/pod_stub.py`; a live run additionally
requires `RUNPOD_LIVE=1` **and** explicit deploy authorization from the
operator (the `T-release` gate never goes live).

## Steps

1. Build the training image from this repo at the pinned commit recorded
   in `artifacts/gates/T-release/release.json` (`commit` field).
2. `create(image)` → `pod_id`. Copy only the dataset adapters and
   manifests the license fence cleared (never firewall `eval-only`
   benchmarks, never secrets — the bundle scan rejects them).
3. `run(pod_id, cmd)` executes the curriculum run; checkpoints stream
   back with their sha256 manifest entries.
4. `destroy(pod_id)` — **unconditional**. The blessed entry point is
   `run_ephemeral(client, image, cmd)`, which destroys in a `finally`
   block, so a failed run cannot leak a billable Pod.

## Verification

- `test_release.py::test_pod_stub_always_destroys` asserts destroy runs
  on success **and** on failure, and that `live_enabled()` is false
  without `RUNPOD_LIVE=1`.
- If a live run ever happens, the gate requires the provider-side destroy
  receipt hash to be recorded in the release bundle; without it the
  release does not close.
