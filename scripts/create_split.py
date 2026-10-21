from __future__ import annotations

import argparse
from pathlib import Path

from binwise.training import create_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/split.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(create_split_manifest(args.dataset_root, args.output, args.seed).resolve())


if __name__ == "__main__":
    main()

