---
id: T-repo-clean
title: Limpiar el repositorio — target/, datasets y pickles fuera del historial
status: backlog
priority: medium
owner: unassigned
category: project
initiative: oss-release
depends_on: []
created: 2026-09-21
updated: 2026-09-21
---

# Limpiar el repositorio — target/, datasets y pickles fuera del historial

Hallazgo H: **9 145 ficheros de `target/`** (build de debug de Rust) están
versionados en git, más datasets y modelos `.pkl` por run (~250 MB entre
huffpost, logiqa, massive y los pickles de cada corrida). Un `git clone` hoy
es inviable y el historial no se puede publicar.

Trabajo:

1. `.gitignore` correcto **primero**: `target/`, `artifacts/runs/`,
   `artifacts/weights/`, `artifacts/data-prefetch/`, `*.pkl`, `.venv-train/`,
   `__pycache__/`, `.DS_Store`.
2. `git rm -r --cached` de lo que ya está trackeado, en un commit único y
   anotado.
3. Reescritura del historial (`git filter-repo`) para sacar los blobs
   grandes. **Operación irreversible**: se hace sobre un clon de trabajo, con
   backup del repo actual verificado antes, y solo con el visto bueno
   explícito del operador. Si el operador prefiere no reescribir, la
   alternativa es un repo público nuevo desde un commit limpio — decidir y
   dejarlo escrito.
4. Los artefactos que sí importan (pesos, datasets grandes, checkpoints)
   pasan a almacenamiento externo con su sha256 registrado en
   `.meshkore/docs/source-register.md`, no a LFS por defecto.
5. Un script `tools/repo_size_check.py` que falle en CI si un commit añade un
   fichero > 5 MB o un directorio ignorado.

## Verification gate

- `git count-objects -vH` del repo resultante < 50 MB.
- `git clone` limpio + `cargo build` reconstruye `target/` sin nada que
  faltara del historial.
- El check de tamaño falla al inyectar a propósito un fichero de 10 MB.
- El gate escribe `artifacts/gates/T-repo-clean/gate.json` con tamaños antes
  y después y el sha del backup.

## Done when

- El backup del repo actual está verificado y su ubicación registrada.
- `git clone` del repo público pesa < 50 MB, sin `target/` ni datasets.
- El CI rechaza commits que vuelvan a meter binarios grandes.
