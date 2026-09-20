---
title: Model card — jev-clone V1 research release
updated: 2026-09-20
owner: general-09192230
---

# Model card — jev-clone V1 (research release)

Research release only (`jev-clone` is the internal name until a commercial
name lands, plan §171). Every claim below links to a raw artifact under
`artifacts/gates/`; the `T-release` gate recomputes the hashes and fails
the release if any claim drifts from its evidence.

## What it is

Decision engine state→distributions: V1 answers `choice` (plus `boolean`
as binary choice) with explicit `unknown` abstention. No text generation.
Served by the same Rust binary locally and in Docker (`T-cloud-api` parity).

## Evidence (raw artifacts)

- Backbone selection: `artifacts/gates/T-bakeoff/report.json` — top-2
  `proxy-L`, `proxy-M` on seeded proxies; real weights
  (`ettin-68m`, `modernbert-base`) still `pending_weights`.
- Gold pilot: `artifacts/gates/T-gold/report.json` — 500 synthetic-seeded
  items (human pilot pending), cohen_kappa 0.0, exact_agree_rate 0.852,
  calibration split 100 / test split 400, leakage clean.
- Calibration: `artifacts/gates/T-calib/calibration.json` —
  predictor `cosine-char3-softmax` v1, global temperature 0.05
  (NLL before 0.5646935979401707, after 0.023480522517238117, n=100),
  per-locale en-US / es-ES; GO criteria in
  `artifacts/gates/T-calib/report.json` (ece_target 0.05,
  accuracy_preserved true).
- Serving: `artifacts/gates/T-cloud-api/coldstart.json` — verdict GO
  against a 30.0 s budget; API contract in `.meshkore/docs/api-v1.md`.
- Bill of materials: `sbom.json` (also baked into the image as
  `/sbom.json`); multi-stage Dockerfile carries no Python.

## Data, licenses, limitations

- Train P0/P1 per `.meshkore/docs/source-register.md`: HuffPost (mirror
  CC0, content rights fenced), Banking77 CC-BY-4.0, BoolQ CC-BY-SA-3.0
  (share-alike obligations apply), Civil Comments CC0-1.0, HelpSteer2
  CC-BY-4.0, MASSIVE CC-BY-4.0 (`massive-pilot-fx1`,
  sha256 `1c1c437e…3c183c1a`). CLINC150/OOS and ANLI stay `eval-only`;
  MMLU-Pro, GPQA, SimpleQA, MuSR, RewardBench 2, ARC, OpenBookQA are
  firewall `eval-only` and never enter training.
- Workspace code is MIT OR Apache-2.0 (declared in each crate manifest;
  backup candidates ModernBERT Apache-2.0, Ettin MIT, NeoBERT MIT,
  LFM2.5 under LFM Open v1.0 — see source-register).
- Limitations: gold labels are synthetic-seeded until the human pilot
  lands (kappa 0.0 — do not quote agreement as human); real-backbone
  weights pending, so quality numbers are proxy-relative, not absolute;
  boolean is binary choice, `score`/extract/multiselect are V2; cascade
  to an LLM fallback is deferred to V2.
- No release ships with an open legal or contamination gate: the
  `T-release` bundle is atomic and any pending gate blocks it.

## Production telemetry

`GET /metrics` exposes Prometheus counters (`jev_requests_total`,
`jev_errors_total`, `jev_cache_hits_total`, `jev_cache_misses_total`,
`jev_infer_us_sum`) — counts and latency sums only, never state content,
weights or option IDs.
