from __future__ import annotations

import argparse
from pathlib import Path

from binwise.retrieval import RuleRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/index"))
    args = parser.parse_args()
    RuleRetriever().save_indexes(args.output)
    print(f"Saved city-separated FAISS indexes to {args.output.resolve()}")


if __name__ == "__main__":
    main()

