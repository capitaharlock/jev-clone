# JevClone Clean-Room Eval Report (JEVALS_REPORT.md)

Reproducible by seed `20260921` + manifest sha `1950bd161960d179c3652fb9dcac317542176f47cd3714aded0dfea11f50fe78` (`eval/data_eval.py`, `eval.json`).

## Clean room

| jevals ids frozen | 5000 |
| clean-room version | 1 |
| ids manifest sha | 7d32e162a762cdcda67b076960a4655586310e90f4c160456452f4a008547ae0 |
| train ids scanned | 131322 |
| canaries scanned | 16 |
| triple detector | exact + normalized + semantic (cos>=0.85) |
| verdict | CLEAN |

## Dynamic labels (unseen B)

| n_B | 7475 |
| zero-shot acc | 0.4385 |
| id-memorization acc | 0.0000 |
| chance | 0.3612 |
| PASS | True |

Memorization of option/label ids scores ~0 on unseen labels B; the zero-shot predictor must clear chance + margin.
