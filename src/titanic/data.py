"""Load the Kaggle Titanic files into one frame (train rows first, then test rows)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def load(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index]:
    """Return (all passengers, train labels, train index, test index)."""
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    df = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)
    train_idx = pd.Index(range(len(train)))
    test_idx = pd.Index(range(len(train), len(df)))
    y = pd.Series(train["Survived"].to_numpy(), index=train_idx, name="Survived")
    return df, y, train_idx, test_idx


def download(raw_dir: Path = RAW) -> None:
    """Fetch the competition files with the Kaggle CLI (needs accepted rules + API key)."""
    import subprocess
    import zipfile

    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "titanic", "-p", str(raw_dir)],
        check=True,
    )
    with zipfile.ZipFile(raw_dir / "titanic.zip") as z:
        z.extractall(raw_dir)
    (raw_dir / "titanic.zip").unlink()


if __name__ == "__main__":
    download()
