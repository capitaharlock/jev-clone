# Calibration Report (CALIBRATION_REPORT.md)

Reproducible by seed `20260921` + manifest sha `1950bd161960d179c3652fb9dcac317542176f47cd3714aded0dfea11f50fe78` (`eval/data_eval.py`, `eval.json`).

Split discipline: temperatures fit on the sealed calibration split ONLY (never gradient-train on it); general + OOD scored once.

| calibration n (real labels) | 20000 |
| domains | boolq, email-triage, logiqa, massive, reclor |
| K values | 2, 4 |
| locales | de-DE, en-US, es-ES, fr-FR, it-IT, pt-PT |
| global temperature | 0.0928 |
| ECE target | 0.05 |
| general ECE post-cal | 0.0254 |

## Section-115 objective (beat baselines before looking at Jev)

| uniform | acc=0.4558 NLL=1.1559 ECE=0.1227 |
| poc_cosine_uncalibrated | acc=0.4558 NLL=1.1353 ECE=0.1127 |
| calibrated | acc=0.4558 NLL=1.0458 ECE=0.0254 |
| objective PASS | True |
