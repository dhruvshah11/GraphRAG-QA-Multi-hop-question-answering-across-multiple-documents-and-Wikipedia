"""Render docs/diagrams/src/*.puml to PNG and SVG with a local PlantUML jar.

    python scripts/render_diagrams.py

Requires Java. PlantUML 1.2023.13 is used because it still runs on Java 8; the
jar is downloaded once into tools/ (from the official GitHub release). Diagrams
use PlantUML's built-in "smetana" layout, so Graphviz is NOT required.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "diagrams" / "src"
OUT = ROOT / "docs" / "diagrams" / "rendered"
JAR = ROOT / "tools" / "plantuml.jar"
JAR_URL = "https://github.com/plantuml/plantuml/releases/download/v1.2023.13/plantuml-1.2023.13.jar"


def main() -> int:
    if shutil.which("java") is None:
        print("Java is required to render PlantUML diagrams.")
        return 1
    if not JAR.exists():
        JAR.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading PlantUML ->", JAR)
        urllib.request.urlretrieve(JAR_URL, JAR)
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.glob("*.puml"))
    failures = 0
    for fmt in ("png", "svg"):
        cmd = ["java", "-Djava.awt.headless=true", "-jar", str(JAR), f"-t{fmt}", "-charset", "UTF-8",
               "-o", str(OUT), *[str(p) for p in files]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            failures += 1
            print(res.stdout, res.stderr)
    for p in sorted(OUT.glob("*.png")):
        print(f"rendered {p.name} ({p.stat().st_size // 1024} KB)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
