# DATA_AUDIT — #T-bigsrc (Tasksource / P3 / DocNLI)

| source | rows acc. | rows rej. | tokens≈ | family | type | lang | license | K-dist |
|---|---|---|---|---|---|---|---|---|
| tasksource-instruct | 6 | 0 | 54 | mixed-discriminative (510 tasks) | Choice 4 / Noul 2 | en | Apache-2.0 | 2:3,3:1,4:1 |
| p3 | 4 | 10 | 31 | multi-QA/NLI/sentiment (anli, arc) | Choice 3 | en | Apache-2.0 | 3:all |
| docnli | 6 | 1 | 92 | NLI long-context | Noul 2 | en | BSD-3-Clause | 2:all |

Component audit (Tasksource aggregates ~510 tasks): component licenses are Apache-2.0/MIT/CC-BY-4.0/CC0 per the upstream task registry; any non-commercial component fails `commercial_clean_ok` and stays out of the commercial branch (§99).

License fence commercial-clean: PASS

Leak check: contamination_hits=0, barrier_blocked=[], jevals_ids_known=0 (no benchmarks/jevals/*_ids.json in repo yet — overlap check is vacuous and documented), jevals_overlap=0, clean=True.

P3 sampler report: {"accepted": 4, "kept_order_seed": 7, "rejected_dataset_cap": 0, "rejected_dupe_input": 10, "rejected_template_cap": 0}.
DocNLI balance: {"no": 3, "yes": 3}.
Shard hashes: tasksource=0f7f6987254d57c4, p3=c11971b6926fa2aa, docnli=12f9359c9bd30c7a.
