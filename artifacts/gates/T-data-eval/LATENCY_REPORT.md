# Latency Report (LATENCY_REPORT.md)

Reproducible by seed `20260921` + manifest sha `1950bd161960d179c3652fb9dcac317542176f47cd3714aded0dfea11f50fe78` (`eval/data_eval.py`, `eval.json`).

Multi-question 1->50 over one shared state: the state encoding is computed once and reused across questions (shared-state) vs recomputed per question (naive).

| Q=1 | naive 0.001s / shared 0.000s / 2.89x / 1.7KB |
| Q=5 | naive 0.004s / shared 0.001s / 4.93x / 2.0KB |
| Q=10 | naive 0.008s / shared 0.001s / 5.33x / 2.9KB |
| Q=25 | naive 0.020s / shared 0.004s / 5.50x / 17.3KB |
| Q=50 | naive 0.039s / shared 0.007s / 5.85x / 17.7KB |
| speedup @50Q (min 1.2) | 5.85 |
| per-question ms @50Q | 0.134 |
| PASS | True |
