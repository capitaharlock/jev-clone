#!/bin/bash
# start-all.sh — arranca TODO el sistema de entrenamiento/generación y lo deja
# trabajando indefinidamente, desacoplado de esta terminal (puedes cerrarla).
#
# Uso:
#   /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone/start-all.sh [PORT]
#
# Arranca:
#   monitor de entrenamiento (tools/training_monitor.py): reentreno completo
#   al detectar datos nuevos + dashboard en http://127.0.0.1:PORT/ .
# El generador Q-W-E-N/programático (tools/data_gen_loop.py) YA NO se arranca
# aquí: #T-halt-contam lo congeló en cuarentena (190 esqueletos, split i%10,
# acc 1.0000 por fuga). Volverá como job del daemon cuando #T-gen-schemas lo
# sustituya. Comprueba al final que el monitor vive, que el dashboard responde
# HTTP 200 y que Ollama (Q-W-E-N) está reachable. Idempotente: mata restos previos.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8794}"
VENV_PY="$ROOT/.venv-train/bin/python"
GENLOG="$ROOT/artifacts/logs/genloop/gen.log"
GENPID="$ROOT/artifacts/logs/genloop/gen.pid"
MONLOG="$ROOT/artifacts/logs/monitor/monitor.log"
MONPID="$ROOT/artifacts/logs/monitor/monitor.pid"

mkdir -p "$ROOT/artifacts/logs/genloop" "$ROOT/artifacts/logs/monitor"

echo "== parando restos previos =="
[ -f "$GENPID" ] && kill "$(cat "$GENPID")" 2>/dev/null && echo "gen pid $(cat "$GENPID") parado" || true
[ -f "$MONPID" ] && kill "$(cat "$MONPID")" 2>/dev/null && echo "monitor pid $(cat "$MONPID") parado" || true
rm -f "$GENPID" "$MONPID"
pkill -f "data_gen_loop.py" 2>/dev/null || true
pkill -f "training_monitor.py" 2>/dev/null || true
if lsof -ti :"$PORT" >/dev/null 2>&1; then kill $(lsof -ti :"$PORT") 2>/dev/null || true; sleep 1; fi
sleep 1

echo "== generador: NO se arranca (congelado por #T-halt-contam, ver cuarentena) =="
# tools/data_gen_loop.py queda fuera hasta que #T-gen-schemas lo sustituya.
# Si el loop debe seguir vivo en alguna forma, que sea como job del daemon.
if pgrep -f "data_gen_loop.py" >/dev/null 2>&1; then
  echo "AVISO: hay un data_gen_loop.py vivo (no lo arrancó este script)"
fi

echo "== arrancando monitor + dashboard :$PORT (desacoplado) =="
nohup "$VENV_PY" "$ROOT/tools/training_monitor.py" --interval 300 --tick 20 \
  --train --port "$PORT" </dev/null >>"$MONLOG" 2>&1 &
echo $! > "$MONPID"
disown %1 2>/dev/null || true
echo "monitor pid $(cat "$MONPID") log $MONLOG"

echo "== verificando (10s) =="
sleep 10
FAIL=0
if pgrep -f "data_gen_loop.py" >/dev/null 2>&1; then
  echo "AVISO generador vivo (no debería: congelado por #T-halt-contam)"
else
  echo "OK generador parado (congelado por #T-halt-contam)"
fi
ps -p "$(cat "$MONPID")" >/dev/null 2>&1 && echo "OK monitor vivo pid $(cat "$MONPID")" || { echo "FAIL monitor muerto, mira $MONLOG"; FAIL=1; }
if "$VENV_PY" -c "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:$PORT/', timeout=10).status == 200" 2>/dev/null; then
  echo "OK dashboard http://127.0.0.1:$PORT/ (HTTP 200)"
else
  echo "FAIL dashboard no responde en :$PORT"; FAIL=1
fi
curl -s -m 5 http://127.0.0.1:11434/api/tags | grep -q "qwen3.6:27b-mlx" \
  && echo "OK Q-W-E-N (qwen3.6:27b-mlx) en Ollama" \
  || { echo "AVISO Ollama sin qwen3.6:27b-mlx: el generador tira solo con plantillas programaticas"; }
echo "synth-loop: congelado en artifacts/data-prefetch/quarantine/ (#T-halt-contam)"
echo "puedes cerrar esta terminal. Logs: tail -f $GENLOG | tail -f $MONLOG"
exit $FAIL
