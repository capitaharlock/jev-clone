# T-antiscale-diag — what carries the fall

Generated 2026-09-22T10:46:04Z · seed 20260922 · `python3 -m tools.diagnose.antiscale`

## Dominant axis

**Axis 4 — calibration vs ranking — 75.4 % of the fall** on the curve this task was asked about (ettin-68m: 88.1 %, modernbert-base: 62.7 %). The next section says how far that survives the other protocol.

The unseen-label number does not fall because the head falls silent. It falls because the ORDER of the options degrades: with `unknown` taken out of the race, the forced decision falls too, and by 1 M it is no longer distinguishable from chance. Runaway abstention (axis 2) is the rest — 24.6 % — and it is real, but forcing a decision does not recover the curve.

| arm | unseen acc 250 k → 1 M | forced decision 250 k → 1 M | abstain 250 k → 1 M | ranking share |
| --- | --- | --- | --- | --- |
| ettin-68m | 0.239362 → 0.049535 | 0.359375 → 0.192154 | 0.430851 → 0.637965 | 88.1 % |
| modernbert-base | 0.194149 → 0.027593 | 0.272274 → 0.167886 | 0.257646 → 0.701795 | 62.7 % |

### Does that survive the other protocol?

Dominant in **3 of 4** (protocol, arm) cells; pooled 56.4 %, range 4.7–88.1 %.

| protocol | ettin-68m | modernbert-base |
| --- | --- | --- |
| trainer_stage_eval | 88.1 % | 62.7 % |
| unseen_labels_gate | 70.1 % | 4.7 % |

the two protocols agree on the direction everywhere and on the SPLIT only in part: they are different cuts of different sizes (the #T-unseen-labels one carries a banking77 sibling-pair holdout the trainer's stage eval does not), so the share is published as a range and not as one decimal

## The five axes

### Axis 1 — label-space memorisation (measured)

the prediction the task wrote down — 'the fall concentrates in the closed-vocabulary cuts' — cannot be falsified on this corpus, because there is no other kind of cut in it. Every source scores as closed vocabulary, including the three whose options are per-row spans: prog-gold 1.0, swag 1.0, synth-v1 0.729297 (the fraction of rows carrying at least one option text the sampler's global pool overwrote). A text -> label map is therefore a sufficient training signal for 100 % of the mixture, which is the mechanism axis 4 measures the consequence of

### Axis 2 — runaway abstention (measured)

abstention rises on every arm of every protocol — it is the only quantity of this decomposition that does — and on the curve this task was asked about it carries 24.6 % of the fall on average. Forcing the decision does NOT recover that curve: the forced-decision number falls with it, which is axis 4. Under the #T-unseen-labels protocol the split moves, and it moves far enough on one arm to change which axis is dominant there: see `dominant.robustness`

### Axis 3 — trainable capacity (**not measured**)

the two short runs are not on disk yet. They are the only two this task may spend and they are running as the daemon job `antiscale-wide`; the axis fills itself in when their summaries land and this module is re-run. Nothing is estimated in the meantime

### Axis 4 — calibration vs ranking (measured)

the order degrades: the forced-decision number falls on both arms and carries 75.4 % of the fall on average. At 1 M it is no longer distinguishable from chance on ettin-68m, modernbert-base (ettin-68m: CI95 upper bound 0.206623 vs chance 0.201924, modernbert-base: CI95 upper bound 0.181664 vs chance 0.201924), and strictly BELOW it on modernbert-base. Temperature alone cannot do that — a mis-scaled but correct ordering keeps its argmax

### Axis 5 — training regime (measured)

not by repetition: both arms run 1.0 epochs over the corpus, so no row is seen twice and the fall is not a memorised row being re-fitted. Early stopping on unseen is a SELECTION control, not a cause — it recovers the fall by keeping an earlier point of the same curve, and how much it recovers depends on the patience it is given

## Cross-check: the release protocol's own cuts

- ettin-68m ALL: fall 0.153983, ranking share 70.1 %
- modernbert-base ALL: fall 0.123755, ranking share 4.7 %
- extreme cell: massive @ modernbert-base abstains on 1.0 of rows at 1 M while ranking 0.218294 vs chance 0.205078

## Priority

**T-gen-objective first, T-unfreeze-backbone second** — decided by axis 4 (calibration vs ranking), 75.4 % of the fall.

- the fall is a RANKING collapse, not a temperature: the forced-decision number falls with the raw one and reaches chance at 1 M. A wrong order is what the objective teaches, and temperature — the thing an unfrozen backbone would not fix either — is the smaller half
- axis 1 measures that 100 % of the 1 M mixture is answerable by a text -> label map: every extra row is another row of that map, so the objective gets exactly what it optimises and the held-out label space is pushed down with it
- the SAME frozen representation supports unseen accuracy well above chance earlier on the curve, so the representation is demonstrably not empty at the point where the curve turns: what changes over the segment is only what the head was taught, since the backbone never updates
- the order does not depend on which of the two partition axes wins. The second protocol puts most of one arm's fall on abstention instead of ranking (see `dominant.robustness`), and runaway abstention is the same kind of artifact: with the backbone frozen, the `unknown` logit is something the head learned under this objective too. Both readings point at the objective before they point at the representation

_#T-unfreeze-backbone is the more expensive intervention and its necessity is exactly what axis 3 tests. Unfreezing a backbone under an objective that is already driving the head the wrong way buys more capacity for the same lesson_

**Flips when:** axis 3 lands: if the 2x or 4x head flattens the 250 k -> 1 M slope, the fall has a capacity component and #T-unfreeze-backbone goes first. Until then this order is the measurement's, not a preference

## Honesty

an axis that cannot be measured from disk is published `measured: false` with its reason. Nothing here is estimated, and no axis is given a share it did not earn from the decomposition above

Gate: `artifacts/gates/T-antiscale-diag/gate.json` · `pass: false` · unmeasured: 3_trainable_capacity
