"""Download and extract RealWaste without committing it to Git."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "realwaste.zip"
    if not archive.exists():
        print(f"Downloading {URL} to {archive} ...")
        with urllib.request.urlopen(URL) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    target = args.output / "realwaste"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)
    print(f"Extracted dataset to {target.resolve()}")
    print("Dataset: UCI RealWaste, DOI 10.24432/C5SS4G, CC BY 4.0")


if __name__ == "__main__":
    main()

