# #T-mix-5m — NO-GO on scaling to 5 M

**Verdict: NO-GO.** No 5 M corpus, no 100 k distillation pilot, no 10 M
curve. Gate artifact: `artifacts/gates/T-mix-5m/gate.json` (`pass: false`).

#T-mix-5m's own verification gate makes the spend conditional — «Requiere
PASS de `T-mix-1m` con ganancia demostrada; si 1 M no mejora, esta task
registra NO-GO y no genera 5 M» (§§89, 136). `T-mix-1m` passed as a *corpus
build*. The other half of the precondition, a demonstrated gain, is not
met. Registering that is the deliverable, not a failure to deliver.

All numbers below are read from the artifacts the `mix1m-curve` job left on
disk (exit 0, `2026-09-22T01:41Z`), and every one of them is reproduced in
`gate.json` so CI can check the decision against the evidence without the
gitignored run directories.

---

## 1. The curve is flat, and where it is not flat it points down

Primary metric, unseen-label accuracy (#T-unseen-labels), measured inside
the *same* run on the *same* corpus by the *same* trainer. Chance 0.165236,
n = 3 008 per stage:

| run | 250 k | 1 M | Δ | chance |
|---|---|---|---|---|
| `mix1m-curve-modernbert-base-s20260922` | 0.194149 | **0.027593** | **−0.166556** | 0.165236 |
| `mix1m-curve-ettin-68m-s20260922` | 0.239362 | **0.049535** | **−0.189827** | 0.165236 |

Both backbones end 1 M *below chance*. The five-point within-run curve shows
this is not a last-stage accident — modernbert-base peaks at 250 k and
collapses from there (62 k 0.080120 → 125 k 0.142620 → 250 k 0.194149 →
500 k 0.132314 → 1 M 0.027593), with abstention climbing 0.257646 → 0.701795
and ECE 0.209980 → 0.619949 over the same stretch. Seen-label accuracy fell
too (0.606051 → 0.573138). The extra 750 k rows moved *confidence*, not
capability.

## 2. Reasoning evals never left chance — at either scale

Eval-only cuts (#T-halt-contam: logiqa/reclor are never training rows).
Options-only accuracy with its CI95, against the cut's own options-only
chance rate. This is `eval/unseen.py`'s test — a point estimate over chance
is not a result; the CI95 *lower* bound has to clear it.

| eval | chance | 250 k (`train-real-v1`) | 1 M modernbert-base | 1 M ettin-68m |
|---|---|---|---|---|
| logiqa-mc (n=1500) | 0.250 | 0.269333 [0.2475, 0.2924] | 0.264000 [0.2423, 0.2869] | 0.238000 [0.2171, 0.2602] |
| reclor (n=500) | 0.250 | 0.212000 [0.1784, 0.24995] | 0.268000 [0.2311, 0.3085] | 0.238000 [0.2028, 0.2772] |
| logiqa-nli (n=1500) | 0.500 | 0.505333 [0.4801, 0.5306] | 0.496667 [0.4714, 0.5219] | 0.492000 [0.4668, 0.5173] |

**Nine cuts out of nine contain their own chance rate inside the interval.**
Not one clears it, at 250 k or at 1 M. There is no gain to extrapolate.

## 3. It does not even fit the reasoning rows it trained on

These are the trained head re-scored over the 1 000 000 rows it was
**trained on** (`summary.json["train"]`) — training scores, not held-out
ones, which makes them the stronger form of the finding:

| dataset | n | modernbert-base | ettin-68m | what it asks for |
|---|---|---|---|---|
| dbpedia14 | 149 900 | 0.971728 | 0.976364 | topic ↔ text overlap |
| civil-comments | 149 900 | 0.942148 | 0.941328 | surface toxicity cues |
| snli | 149 900 | 0.761254 | 0.753442 | lexical entailment cues |
| **prog-gold** | 60 631 | **0.441094** | **0.440484** | grounded lookup |
| **swag** (K = 4, chance 0.25) | 73 546 | **0.255024** | **0.256601** | commonsense continuation |
| **synth-v1** | 34 847 | **0.229747** | **0.235630** | generated decisions |

The split is the whole story: where the answer is *in* the text, the model
is near ceiling; where the answer has to be *inferred* from it, the model
cannot beat a coin even by memorising. **It matches text; it does not
reason.** More rows of the same kind multiply what it already does.

## 4. Why — the hypothesis the curve supports

`run.json` says `backbone.frozen: true` for every run on this curve. The
trainable surface is a pointer head over an encoder that never updates:

| run | frozen backbone | trainable head | trainable |
|---|---|---|---|
| modernbert-base | 149 014 272 | 2 896 386 | **1.91 %** |
| ettin-68m | 68 144 640 | 2 699 778 | **3.81 %** |

(head sizes measured by #T-bakeoff-real on this exact head architecture,
`artifacts/gates/T-bakeoff/report.json`.)

A frozen encoder's representation is fixed before the first row arrives.
The head can learn to *route* within that representation — which is exactly
the text-matching skill that is at ceiling — but it cannot add a capability
the representation does not already carry. Under that reading the flat
reasoning curve is not surprising and not fixable with data: rows 1 M
through 5 M are more evidence for a function that is already fit.

This is the hypothesis the curve **supports**. It is not proven: no run on
this curve unfroze the backbone, so the experiment that would separate
"data is the bottleneck" from "the frozen representation is the bottleneck"
has not been run. Running it is the cheap next step; 5 M is the expensive
one.

## 5. What would flip this to GO

All three, in order:

1. An **unfrozen** (or LoRA / partially-unfrozen) backbone run at **250 k**
   on this same corpus clears chance on at least one eval-only reasoning cut
   by CI95 lower bound. 250 k, not 1 M — if the bottleneck is the frozen
   representation, the signal shows at the cheap scale.
2. Under that configuration, the 250 k → 1 M delta on unseen-label accuracy
   is **positive** for at least one backbone. A non-flat curve is what §136
   asks for; a higher intercept alone is not.
3. The gain survives the #T-unseen-labels protocol unchanged — same
   holdout, same calibration fitted on seen labels only. A gain that needs
   the protocol relaxed is not a gain.

Until then `gate.json` stays `pass: false`, and
`tools/mix_5m/nogo.py::violations` (tested by `data/test_mix_5m.py`) refuses
any gate that flips to GO while the reasoning evals sit at chance.

---

## Corrections to the numbers as first reported

Small, and none of them changes the verdict:

* the pointer head is **2.90 M** parameters (modernbert-base) / 2.70 M
  (ettin-68m), not ~1.6 M — 1.91 % / 3.81 % of the total, measured;
* the headline per-dataset figures (swag 0.255, synth-v1 0.230, prog-gold
  0.441, dbpedia14 0.972, civil-comments 0.942) are the **training-corpus
  re-score**, not a held-out eval. They are right, and they mean *more*
  than reported: that is the model failing on rows it saw;
* dbpedia14 is 0.971728 (modernbert-base), 0.976364 (ettin-68m).
