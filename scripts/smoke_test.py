"""Run 20 held-out images and report local inference latency."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from PIL import Image

from binwise.classifier import VisionClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/split.csv"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(rows) < args.runs:
        raise ValueError(f"The manifest contains only {len(rows)} test images")
    classifier = VisionClassifier(args.weights)
    latencies = []
    for row in rows[: args.runs]:
        with Image.open(row["path"]) as image:
            started = time.perf_counter()
            predictions = classifier.predict(image)
            latencies.append(time.perf_counter() - started)
        print(f"{Path(row['path']).name}: {predictions[0].label} ({predictions[0].confidence:.1%})")
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    print(f"Runs: {len(latencies)}")
    print(f"Median: {statistics.median(latencies):.3f}s")
    print(f"p95 (local, queue excluded): {ordered[p95_index]:.3f}s")


if __name__ == "__main__":
    main()

