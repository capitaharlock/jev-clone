# Generalization Report (GENERALIZATION_REPORT.md)

Reproducible by seed `20260921` + manifest sha `1950bd161960d179c3652fb9dcac317542176f47cd3714aded0dfea11f50fe78` (`eval/data_eval.py`, `eval.json`).

## JevClone-General-Eval (v1)

| n held-out | 10000 |
| accuracy | 0.4558 |
| NLL | 1.0458 |
| Brier | 0.6009 |
| ECE | 0.0254 |
| high-conf errors (>=0.9) | 6 (0.0006) |

## Stability battery

| option-shuffle flip rate | 0.0000 (max 0.02) |
| option-shuffle JS mean | 0.000000 (max 0.01) |
| candidate-insertion stability | 1.0000 (min 0.95) |
| risk-coverage beats random | True |
| avg selective risk vs overall err | 0.3932 vs 0.5442 |

## OOD / abstention

| in-domain mean conf | 0.4702 |
| OOD mean conf | 0.3053 |
| OOD abstention rate | 0.3009 |
| in-domain abstention rate | 0.0000 |
| PASS | True |
