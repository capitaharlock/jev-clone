#!/usr/bin/env python3
"""Generate sbom.json from Cargo.lock (#T-cloud-api).

No invented versions: every entry comes from the lockfile. The image
copies this file to /sbom.json so the release can be audited.
Usage: python3 tools/sbom.py [--check]  (run from repo root)
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "Cargo.lock")
OUT = os.path.join(ROOT, "sbom.json")


def parse_lock(path: str) -> list:
    text = open(path).read()
    pkgs = []
    for chunk in text.split("[[package]]")[1:]:
        name = re.search(r'^name = "(.*)"', chunk, re.M).group(1)
        version = re.search(r'^version = "(.*)"', chunk, re.M).group(1)
        pkgs.append({"name": name, "version": version})
    return sorted(pkgs, key=lambda p: (p["name"], p["version"]))


def main() -> int:
    if not os.path.isfile(LOCK):
        print("no Cargo.lock; build once first", file=sys.stderr)
        return 1
    sbom = {"sbom": "jevclone", "source": "Cargo.lock", "packages": parse_lock(LOCK)}
    if "--check" in sys.argv:
        disk = json.load(open(OUT))
        ok = disk == sbom
        print("SBOM " + ("in sync" if ok else "STALE"))
        return 0 if ok else 1
    json.dump(sbom, open(OUT, "w"), indent=2)
    print(f"wrote {OUT} ({len(sbom['packages'])} packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
