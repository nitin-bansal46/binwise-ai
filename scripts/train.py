from __future__ import annotations

import argparse
from pathlib import Path

from binwise.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/split.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    metrics = train(
        args.manifest,
        args.output,
        args.batch_size,
        args.workers,
        resume_checkpoint=args.resume,
    )
    print(f"Test macro-F1: {metrics['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

