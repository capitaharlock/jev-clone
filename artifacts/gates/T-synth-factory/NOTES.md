# T-synth-factory — pilot audit notes

Pilot: `python3 tools/data_factory/run_pilot.py --target 50000` → 50 002 accepted,
`pass: true` in `gate.json` (elapsed 54.7 s, deterministic local teachers).

- Agreement proposer/solver: 0.799 (design ~0.80); adjudicated on disagreement: 6 840.
- Role inversion: 0.500 exactly (odd `seq` inverts Qwen↔DeepSeek).
- Reject rate: 0.153 (validator + exact/semantic dedupe).
- Splits by parent group: train 34 847 / calibration 10 216 / test 4 939.
- Parents: 8 197. Kinds: boolean 16 456 / choice 33 546. Zero CoT keys stored.
- Source limitation (pilot, not a failure): target was met inside the first
  shard in order (`boolq`), so all 50 k are grounded on BoolQ states. Multi-source
  mixing is deferred to `#T-mix-1m` / `#T-prog-gold`.

## Upstream finding: boolq question ids are not unique across rows

Every row of `artifacts/data-prefetch/boolq.jsonl` carries question id `boolq-0`.
First pilot run collapsed to 1 parent / single split because of it.
`source_streamer.py` now derives `parent_example_id` from `source + sha256(state)`
and no longer trusts row question ids. The boolq adapter itself is untouched
(done task, shards pinned) — flagged here for a future adapter revision.
