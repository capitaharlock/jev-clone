"""RunPod Pods train-flow stub (#T-release).

Live Pod training is opt-in (``RUNPOD_LIVE=1`` + deploy authorization) and
never runs in gates. This stub models the contract the live flow must obey:

- a Pod is created, the run executes, the Pod is **destroyed afterwards**,
  even when the run fails;
- :func:`run_ephemeral` is the only entry point the docs bless, so the
  destroy step cannot be skipped by accident.
"""
from __future__ import annotations


class FakePodClient:
    """In-memory stand-in for the RunPod Pods API."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.destroyed: list[str] = []
        self._seq = 0

    def create(self, image: str) -> str:
        self._seq += 1
        pod_id = f"pod-stub-{self._seq}"
        self.created.append(pod_id)
        return pod_id

    def run(self, pod_id: str, cmd: str) -> str:
        if cmd == "fail":
            raise RuntimeError("stub run failed")
        return f"ok:{pod_id}:{cmd}"

    def destroy(self, pod_id: str) -> None:
        self.destroyed.append(pod_id)


def run_ephemeral(client: FakePodClient, image: str, cmd: str) -> str:
    """Create → run → destroy. Destroy always executes (try/finally)."""
    pod_id = client.create(image)
    try:
        return client.run(pod_id, cmd)
    finally:
        client.destroy(pod_id)


def live_enabled() -> bool:
    """Live runs require explicit opt-in; gates always take the stub path."""
    import os

    return os.environ.get("RUNPOD_LIVE") == "1"
