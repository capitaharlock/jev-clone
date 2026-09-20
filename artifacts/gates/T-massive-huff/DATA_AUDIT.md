# DATA_AUDIT — #T-massive-huff (MASSIVE 6 locales, HuffPost completo, LogiQA/ReClor)

| source | rows in | rows acc. | rows rej. | reason for delta | splits (tr/cal/te) | license |
|---|---|---|---|---|---|---|
| MASSIVE (6 locales) | 99 126 | 99 126 | 0 | — (tarball stream en-US/es-ES/fr-FR/de-DE/pt-PT/it-IT) | 69084/12198/17844 | Apache-2.0 (Amazon) |
| HuffPost (41 cats) | 200 853 | 181 140 | 19 713 | empty headline/description in source; **0 unknown categories** | 162935/0/18205 | gated: content-rights fence (research until §148) |
| LogiQA 2.0 | 84 976 | 84 935 | 41 | 4 truncated inner-JSON dump lines; 37 empty passage/question/option | 63683/10907/10345 | research (derived from published set) |
| ReClor | 6 138 | 6 138 | 0 | test split (1000) has hidden gold → `answer: unknown`, never trains | 4638/500/1000 | research (derived from published set) |

## HuffPost 66 k vs 210 k — resolved (§§31, 125)

The old adapter pinned `HUFFPOST_UNIVERSE` to 8 categories, so 134 343 of
200 853 rows were rejected as "unknown category". The snapshot actually
carries **41** distinct categories. The universe is now all 41 (legacy
FOOD kept as 42nd harmless distractor; the 8 originals stay first so
seeded draws on those labels are stable). Remaining delta (19 713) is
100 % degenerate source rows (empty headline or description), verified by
re-scanning the snapshot: 0 rows with a category outside the universe.

## Layer A frozen (§61)

Human/expert-label sources now closed: boolq (12 697) + civil-comments
(1 999 514, cap enforced per-cycle §§27/126) + massive-6locale (99 126) +
huffpost-full (181 140) + logiqa (84 935) + reclor-train/cal (5 138) +
banking77 (13 083, eval-side) + helpsteer2 (21 362, eval-side).
ReClor-test (1000, hidden gold) and NLI/zero-shot/DPO outputs stay in
`research-only/` until provenance audit (§§23–24).

## Civil cap probe (§§65–66)

`civil_share` counter verified both ways: capped probe mix (150/1000 =
15 %) passes; uncapped mix (900/1000 = 90 %) fails. Gate fails any mix
with civil > 15 %.

## License matrix (fuente × licencia × split)

| source | license | train | commercial branch |
|---|---|---|---|
| massive-6locale | Apache-2.0 | ok | ok |
| huffpost-full | gated (content rights) | fence-blocked | blocked (§99, §148) |
| boolq | CC-BY-SA-3.0 | fence-blocked | share-alike obligation documented |
| logiqa / reclor | derived-research | ok (research) | research-only until audit |
| civil-comments | CC0-ish (Jigsaw) | ok, capped 15 %/mix | ok, capped |

## No-explanation guarantee (§124)

LogiQA reasoning-type flags dropped; ReClor stores no explanations.
Converter scans serialized **field names** (not prose — the word
"explanation" occurs legitimately in passages) and fails on any
explanation-like field. Tests assert gold correctness + field scan.

## Repro

- `python3 data/convert_huffpost.py` → `artifacts/data-prefetch/huffpost.jsonl`
- `python3 data/convert_massive.py` → `artifacts/data-prefetch/massive.jsonl`
- `python3 data/convert_logiqa_reclor.py` → `logiqa.jsonl` + `reclor.jsonl`
- `python3 -m unittest data.test_massive_huff` (17 tests)
- `artifacts/data-prefetch/manifest.json` pins rows/splits/sha per job.
