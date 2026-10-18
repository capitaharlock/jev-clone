# Decision API V1 (`#T-cloud-api`)

Public decision API. V1 covers `choice` (any N options; `boolean` is a
2-option choice) plus `unknown` by abstention threshold. `score` is
deferred to V2.

Base: `http://HOST:8080`. Every response echoes `schema: "v1"` and the
`model_version`; option order and IDs are preserved.

## `POST /v1/choice` — state → distributions

```json
{
  "model_version": "m1",
  "options": [{"id": "yes"}, {"id": "no"}],
  "weights": [0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
  "bias": [0.1, -0.1],
  "input": [1.0, 0.5, -0.5, 0.25],
  "threshold": 0.5
}
```

```json
{
  "schema": "v1",
  "model_version": "m1",
  "distribution": [{"id": "yes", "prob": 0.62}, {"id": "no", "prob": 0.38}],
  "choice": "yes",
  "unknown": false,
  "cached": false
}
```

`threshold` (default 0.5, range [0,1]): the top option wins only if its
probability reaches it, otherwise `choice` is `null` and `unknown` is
`true`. `threshold: 1.0` forces abstention.

## `POST /v1/batch` — multi-question

```json
{"queries": [<choice>, <choice>, ...]}
```

Up to 64 queries; each result is `{"status": "ok", <choice...>}` or
`{"status": "err", "code": ..., "message": ...}` — one bad query never
fails the batch, and order matches the input.

## Errors

`{"code": ..., "message": ...}` with HTTP 422 (`bad-schema`,
`bad-version`, `bad-options`, `bad-vector`, `bad-threshold`,
`bad-batch`), 413 (`too-large`), 400 (`bad-envelope`), 408
(`timeout`). Error bodies never echo state (no weights/inputs/hashes).

## Ops

- `GET /health` → `"ok"` (liveness).
- `GET /ready` → `{"ready": true, "device": "cpu", "schema": "v1"}`.
- `POST /runpod` — RunPod Serverless envelope
  (`{"input": <choice>, "id": <jobId>}` → `{"output": ..., "id": ...}`).
  Thin adapter only; live validation needs credentials.

Limits: 128 options, 64 queries/batch, 2^20 vector length, 30 s per
query. Contract tests: `training/python/test_cloud_api.py` (same golden
corpus against local binary and Docker image).
