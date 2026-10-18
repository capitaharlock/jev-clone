"""Bucle sin fin (#T-loop-nightly): mezclar, entrenar, evaluar, promover.

Proceso B de `docs/bucle-infinito.md`. Reanudable: `artifacts/loop/state.json`
dice el ciclo y el escalon. Un fallo no promueve. Sin datos: `starved`.

    PYTHONPATH=. .venv-train/bin/python -u -m training.python.loop run --forever
    PYTHONPATH=. .venv-train/bin/python -m training.python.loop status
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_PATH = ROOT / "artifacts" / "loop" / "state.json"
CYCLE_ROOT = ROOT / "artifacts" / "loop"
TRAINING = CYCLE_ROOT / "TRAINING"
CURRENT = ROOT / "artifacts" / "checkpoints" / "ce" / "current"
PILOT = ROOT / "artifacts" / "episodes-qwen" / "pilot-2k" / "verified.jsonl"
BATTERY = ROOT / "data" / "battery_dev.jsonl"
LADDER = (2000, 5000, 10000, 20000, 50000, 100000)
MINILM_WEIGHTS = "minilmv2-l6-mnli-xnli"
MDEBERTA_WEIGHTS = "mdeberta-v3-base-xnli"
DEFAULT_WEIGHTS = "auto"
BACKBONE_STREAK = 3


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "cycle": 0,
        "step": 0,
        "streak_no_promote": 0,
        "current": None,
        "starved": False,
        "updated_utc": utcnow(),
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = utcnow()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATE_PATH)


def budget_of(step: int) -> int:
    if step < 0:
        step = 0
    if step >= len(LADDER):
        return LADDER[-1] * (2 ** (step - len(LADDER) + 1))
    return LADDER[step]


def select_weights(state: dict, requested: str,
                   present=None) -> str:
    """Choose the sticky backbone for this phase of the loop."""
    if requested != "auto":
        return requested
    active = state.get("active_weights")
    if active and active != MINILM_WEIGHTS:
        return str(active)
    if int(state.get("streak_no_promote") or 0) < BACKBONE_STREAK:
        return MINILM_WEIGHTS
    if present is None:
        from model.weights import _present
        present = _present
    return (MDEBERTA_WEIGHTS if present(MDEBERTA_WEIGHTS)
            else MINILM_WEIGHTS)


def current_eval_path(weights: str) -> Path:
    return CURRENT.parent / f"current-eval-{weights}.json"


def untuned_eval_path(weights: str) -> Path:
    return CURRENT.parent / f"untuned-eval-{weights}.json"


def activate_weights(state: dict, weights: str) -> bool:
    """Start a new independent ladder when the backbone changes."""
    previous = state.get("active_weights") or MINILM_WEIGHTS
    state["active_weights"] = weights
    if previous == weights:
        return False
    state["previous_weights"] = previous
    state["current"] = None
    state["resume"] = None
    state["resume_forced"] = None
    # The larger backbone starts with the 5k smoke required by
    # #T-backbone-ladder, not another 2k MiniLM cycle.
    state["step"] = 1 if weights == MDEBERTA_WEIGHTS else 0
    state["streak_no_promote"] = 0
    return True


def select_device(requested: str, weights: str) -> str:
    """Keep the requested device. The mDeBERTa MPS stall was a 51 GB batch,
    capped inside the trainer, not a reason to leave the GPU."""
    return requested


def init_checkpoint(state: dict) -> str:
    """Continue from the best run of this backbone, else the promoted one."""
    resume = state.get("resume")
    if resume and Path(resume).exists():
        return str(resume)
    current = state.get("current")
    if current and Path(current).exists():
        return str(current)
    return "none"


def keep_resume(state: dict, run: str, forced: float) -> None:
    """A gate can say no and the weights still keep what they learned."""
    best = state.get("resume_forced")
    if best is None or forced >= float(best):
        state["resume"] = run
        state["resume_forced"] = forced


BATTERY_LAYOUT = "battery"


def adopt_battery_layout(state: dict) -> bool:
    """Train on the dev-battery shape: decisive numbers live in the state.

    The published rule batches put those numbers inside the options, and
    a checkpoint trained that way does not carry over.
    """
    if state.get("data_layout") == BATTERY_LAYOUT:
        return False
    state["data_layout"] = BATTERY_LAYOUT
    state["resume"] = None
    state["resume_forced"] = None
    state["current"] = None
    return True


def write_battery_rules(out: Path, n: int, seed: int) -> Path:
    from data import episode_gen as EG
    from data import rule_variety as RV
    EG.run(
        n=n, seed=seed, prose="local", teacher="stub",
        out_dir=str(out), families=list(RV.FAMILIES), variety=16,
        group_prefix=f"bat{seed}-", layout=BATTERY_LAYOUT,
        command=f"battery-layout n={n} seed={seed}",
    )
    return out


def latest_rule_dir() -> Path | None:
    from data import stream as ST
    batches = ST.list_rule_batches()
    if not batches:
        return None
    return ST.RULE_ROOT / f"batch-{batches[-1]:04d}"


def promote_rule(control: dict, tuned: dict) -> dict:
    """Regla del manual §3 sobre las cifras ya publicadas (sin re-bootstrap).

    GO si forzada y contrafactual suben, y ninguna familia ni idioma tiene
    `below_control`. El bootstrap pareado lo firma `ce_finetune gate`;
    aqui se aplica la misma logica de caida por familia/idioma que ese gate
    ya publico en `drops` cuando el control es el current anterior.
    """
    cf = control["figures"]["forced"]["accuracy"]
    tf = tuned["figures"]["forced"]["accuracy"]
    cj = control["figures"]["counterfactual_joint"]["accuracy"]
    tj = tuned["figures"]["counterfactual_joint"]["accuracy"]
    forced_up = tf > cf
    joint_up = tj > cj
    fam_c = {r["family"] for r in control.get("rows") or []}
    # Prefer the gate's by_family if present (compare() output).
    drops = []
    c_fam = (control["figures"].get("macro_by_family") or {}).get("by_family") or {}
    t_fam = (tuned["figures"].get("macro_by_family") or {}).get("by_family") or {}
    for name, cell in t_fam.items():
        prev = (c_fam.get(name) or {}).get("accuracy")
        acc = cell.get("accuracy")
        if prev is not None and acc is not None and acc < prev:
            drops.append(f"family:{name}")
    c_lang = control["figures"].get("by_lang") or {}
    t_lang = tuned["figures"].get("by_lang") or {}
    for name, cell in t_lang.items():
        prev = (c_lang.get(name) or {}).get("accuracy")
        acc = cell.get("accuracy")
        if prev is not None and acc is not None and acc < prev:
            drops.append(f"lang:{name}")
    go = forced_up and joint_up and not drops
    return {
        "forced_up": forced_up,
        "joint_up": joint_up,
        "drops": drops,
        "promote": go,
        "reason": ("GO" if go else
                   f"no-promote forced {cf}->{tf} joint {cj}->{tj} drops={drops}"),
        "families_seen": sorted(fam_c),
    }


def verdict_from_gate(gate: dict) -> dict:
    """Use the signed paired-bootstrap gate as the promotion authority."""
    comparison = gate.get("comparison") or gate
    return {
        "forced_up": comparison["forced"]["point"] > 0,
        "joint_up": comparison["counterfactual_joint"]["point"] > 0,
        "drops": list(comparison.get("drops") or []),
        "promote": bool(gate.get("pass")),
        "reason": str(gate.get("reason") or gate.get("verdict") or "no verdict"),
        "gate_verdict": gate.get("verdict"),
    }


def _py() -> str:
    venv = ROOT / ".venv-train" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _run(argv: list[str]) -> int:
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("[loop]", " ".join(argv), flush=True)
    return subprocess.call(argv, cwd=str(ROOT), env=env)


def set_training(on: bool) -> None:
    TRAINING.parent.mkdir(parents=True, exist_ok=True)
    if on:
        TRAINING.write_text(utcnow() + "\n")
    elif TRAINING.exists():
        TRAINING.unlink()


def point_current(run: Path, weights: str) -> None:
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    marker = CURRENT.with_suffix(".json")
    marker.write_text(json.dumps({
        "run": str(run),
        "updated_utc": utcnow(),
    }, indent=2) + "\n")
    # Soft copy of the eval so status can read it without the run dir.
    ev = run / "eval-dev.json"
    if ev.exists():
        shutil.copy2(ev, current_eval_path(weights))


def one_cycle(state: dict, device: str, weights: str) -> dict:
    weights = select_weights(state, weights)
    device = select_device(device, weights)
    switched = activate_weights(state, weights)
    device_changed = state.get("active_device") != device
    state["active_device"] = device
    if switched or device_changed:
        save_state(state)
    cycle = int(state["cycle"]) + 1
    step = int(state["step"])
    budget = budget_of(step)
    cdir = CYCLE_ROOT / f"cycle-{cycle:04d}"
    cdir.mkdir(parents=True, exist_ok=True)
    if adopt_battery_layout(state):
        save_state(state)
    rule = write_battery_rules(cdir / "battery-rules", budget,
                               20260927 + cycle)
    from model.weights import _present
    if not _present(weights):
        decision = {
            "cycle": cycle, "starved": True,
            "reason": f"weights {weights} not on disk (HF blocked on this net)",
            "promote": False, "waiting_for_weights": True,
        }
        (cdir / "decision.json").write_text(
            json.dumps(decision, indent=2) + "\n")
        state["cycle"] = cycle
        state["starved"] = True
        state["last_decision"] = decision["reason"]
        save_state(state)
        return decision
    if not PILOT.exists():
        decision = {
            "cycle": cycle, "starved": True,
            "reason": "no rule batch or no pilot",
            "promote": False,
        }
        (cdir / "decision.json").write_text(
            json.dumps(decision, indent=2) + "\n")
        state["cycle"] = cycle
        state["starved"] = True
        save_state(state)
        return decision

    mix_out = cdir / "mix"
    py = _py()
    rc = _run([
        py, "-m", "eval.numeric_value", "mix",
        "--rule", str(rule),
        "--pilot", str(PILOT),
        "--n-rule", str(budget),
        "--out", str(mix_out),
    ])
    if rc != 0:
        raise RuntimeError(f"mix failed {rc}")

    init = init_checkpoint(state)
    run_out = ROOT / "artifacts" / "checkpoints" / "ce" / f"loop-{cycle:04d}"
    set_training(True)
    try:
        rc = _run([
            py, "-m", "training.python.ce_finetune", "train",
            "--episodes", str(mix_out),
            "--weights", weights,
            "--init", init,
            "--out", str(run_out),
            "--device", device,
            "--budget", str(budget),
            "--eval-every", str(max(budget // 2, 250)),
            "--holdout", "0.2",
            "--split-seed", "20260927",
        ])
        if rc != 0:
            raise RuntimeError(f"train failed {rc}")
        eval_out = run_out / "eval-dev.json"
        rc = _run([
            py, "-m", "training.python.ce_finetune", "eval",
            "--checkpoint", str(run_out),
            "--weights", weights,
            "--battery", str(BATTERY),
            "--device", device,
            "--out", str(eval_out),
        ])
        if rc != 0:
            raise RuntimeError(f"eval failed {rc}")
    finally:
        set_training(False)

    tuned = json.loads((run_out / "eval-dev.json").read_text())
    control_path = (current_eval_path(weights) if state.get("current")
                    else untuned_eval_path(weights))
    if not control_path.exists():
        # First cycle for this backbone: measure and cache its own control.
        none_out = untuned_eval_path(weights)
        _run([
            py, "-m", "training.python.ce_finetune", "eval",
            "--checkpoint", "none",
            "--weights", weights,
            "--battery", str(BATTERY),
            "--device", device,
            "--out", str(none_out),
        ])
        control = json.loads(none_out.read_text()) if none_out.exists() else tuned
        gate_control = none_out if none_out.exists() else eval_out
    else:
        control = json.loads(control_path.read_text())
        gate_control = control_path

    gate_out = cdir / "gate.json"
    _run([
        py, "-m", "training.python.ce_finetune", "gate",
        "--run", str(run_out),
        "--control", str(gate_control),
        "--out", str(gate_out),
    ])
    gate = json.loads(gate_out.read_text())
    verdict = verdict_from_gate(gate)
    decision = {
        "cycle": cycle,
        "budget": budget,
        "step": step,
        "weights": weights,
        "backbone_switched": switched,
        "init": init,
        "run": str(run_out),
        "rule_dir": str(rule),
        "starved": False,
        **verdict,
        "measured_utc": utcnow(),
    }
    (cdir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    shutil.copy2(run_out / "eval-dev.json", cdir / "eval-dev.json")

    if verdict["promote"]:
        point_current(run_out, weights)
        state["current"] = str(run_out)
        state["streak_no_promote"] = 0
        state["step"] = step + 1
    else:
        state["streak_no_promote"] = int(state.get("streak_no_promote") or 0) + 1
    keep_resume(state, str(run_out),
                float(tuned["figures"]["forced"]["accuracy"]))
    state["cycle"] = cycle
    state["starved"] = False
    state["last_decision"] = decision["reason"]
    save_state(state)
    print(f"[loop] cycle {cycle} promote={verdict['promote']} "
          f"{verdict['reason']}", flush=True)
    return decision


def run_forever(device: str, weights: str, sleep_s: int) -> None:
    print(f"[loop] start device={device} weights={weights}", flush=True)
    while True:
        state = load_state()
        try:
            decision = one_cycle(state, device, weights)
        except Exception as e:
            set_training(False)
            err = {"error": f"{type(e).__name__}: {e}", "utc": utcnow()}
            print(f"[loop] {err}", flush=True)
            (CYCLE_ROOT / "last_error.json").write_text(
                json.dumps(err, indent=2) + "\n")
            time.sleep(sleep_s)
            continue
        wait = 15 if decision.get("starved") else sleep_s
        time.sleep(wait)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.python.loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--forever", action="store_true")
    r.add_argument("--device", default="mps")
    r.add_argument("--weights", default=DEFAULT_WEIGHTS)
    r.add_argument("--sleep", type=int, default=30)
    r.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(load_state(), indent=2, ensure_ascii=False))
        return 0
    if args.once:
        print(json.dumps(one_cycle(load_state(), args.device, args.weights),
                         indent=2, ensure_ascii=False))
        return 0
    if not args.forever:
        print("refusing: pass --forever or --once", file=sys.stderr)
        return 2
    run_forever(args.device, args.weights, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
