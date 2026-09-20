# /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone

This is a [MeshKore](https://meshkore.com/standard) cluster — a local-first multi-agent project. Its entire state lives in this repo's `.meshkore/` ledger; the project carries **no daemon code of its own** — clone it anywhere and it stays portable.

One MeshKore daemon per machine serves every project on that machine, routing each request to this cluster by its id (`users-ricartjuncadella-documents-prj-asi`), not by port. To work on this project, open the cockpit at <https://architect.meshkore.com> and add it by its path — the running daemon adopts it and the cockpit auto-detects it. If no daemon is running on this machine yet, install MeshKore once (see <https://meshkore.com/standard>) and it will serve this and all your other clusters.
