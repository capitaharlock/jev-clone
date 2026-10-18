#!/bin/bash
# Arranca productor + bucle + dashboard desacoplados (24 h).
# Uso: ./tools/run_24h.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-train/bin/python"
export PYTHONPATH="$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

if [ ! -x "$PY" ]; then
  echo "FAIL: falta $PY"
  exit 1
fi

echo "== pesos MiniLM (hf.co suele estar bloqueado en PwC; si falla, el bucle espera) =="
JEV_ALLOW_DOWNLOAD=1 "$PY" -m model.weights fetch minilmv2-l6-mnli-xnli || true
JEV_ALLOW_DOWNLOAD=1 "$PY" -m model.weights verify minilmv2-l6-mnli-xnli || true

echo "== lote por regla 1 si falta =="
if [ ! -f artifacts/episodes-rule/batch-0001/episodes.jsonl ]; then
  "$PY" -u -m data.rule_variety publish --batch 1 --n 5000 --variety 16
fi

"$PY" "$ROOT/tools/spawn_24h.py"
echo "status: $PY -m training.python.loop status"
echo "        $PY -m data.stream status"
