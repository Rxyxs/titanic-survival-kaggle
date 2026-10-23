"""Submit prediction files with the Kaggle CLI and record the scores.

    python -m titanic.submit gender woman_boy_rule ...   # submit, then record scores
    python -m titanic.submit --scores-only               # just refresh reports/leaderboard.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .data import ROOT

SUBS = ROOT / "submissions"
OUT = ROOT / "reports" / "leaderboard.json"


def kaggle() -> str:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    return exe


def submit(name: str) -> None:
    path = SUBS / f"{name}.csv"
    rows = path.read_text().strip().splitlines()
    assert rows[0] == "PassengerId,Survived" and len(rows) == 419, f"bad file {path}"
    subprocess.run([kaggle(), "competitions", "submit", "-c", "titanic", "-f", str(path),
                    "-m", name], check=True)


def scores() -> list[dict]:
    out = subprocess.run([kaggle(), "competitions", "submissions", "-c", "titanic", "--csv"],
                         check=True, capture_output=True, text=True).stdout
    start = out.find("fileName")
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    return [{"name": r["description"], "date": r["date"], "status": r["status"],
             "public_score": r["publicScore"], "private_score": r.get("privateScore", "")}
            for r in rows]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--scores-only", action="store_true")
    a = ap.parse_args()
    if not a.scores_only:
        for n in a.names:
            submit(n)
        time.sleep(20)
    rows = scores()
    OUT.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
