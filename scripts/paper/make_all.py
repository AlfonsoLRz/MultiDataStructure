#!/usr/bin/env python3
"""Regenerate every table, macro and figure of the manuscript from the local result copies.

    .venv-mds/Scripts/python.exe scripts/paper/make_all.py        (Windows; the environment with matplotlib)
    python3 scripts/paper/make_all.py                              (elsewhere)

Runs tables.py, macros.py and figures.py with this interpreter and stops at the first failure, so a
manuscript build can depend on it: `latexmk` reads paper/generated/*.tex and paper/figures/*.pdf.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("tables.py", "macros.py", "figures.py"):
    print(f"== {script}", flush=True)
    code = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE.parent.parent).returncode
    if code:
        sys.exit(f"{script} failed with exit code {code}")
print("== done: paper/generated and paper/figures are current")
