"""Exercise the deployed Space with one image, confirmation, and correction.

Requires ``pip install gradio_client==1.13.3 pillow``. Without --image this uses
a synthetic image to check deployment plumbing, not classification accuracy.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from gradio_client import Client, handle_file
from huggingface_hub import get_token
from PIL import Image


def check_response(result: tuple, *, expected_rule: str | None = None) -> None:
    markdown, _confirmation, _override, trace = result
    if "Could not complete" in markdown:
        raise RuntimeError(markdown)
    print(markdown, flush=True)
    print("TRACE", json.dumps(trace, ensure_ascii=False), flush=True)
    if expected_rule and expected_rule not in markdown:
        raise AssertionError(f"Expected sourced rule {expected_rule}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="nitinbansal46/binwise-ai")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    client = Client(args.space, hf_token=get_token())
    with tempfile.TemporaryDirectory(prefix="binwise-verification-") as temporary:
        image_path = args.image
        if image_path is None:
            image_path = Path(temporary) / "synthetic-check.png"
            Image.new("RGB", (224, 224), (45, 120, 75)).save(image_path)
            print("Synthetic input: deployment check only, not an accuracy test.", flush=True)

        started = time.perf_counter()
        job = client.submit(handle_file(str(image_path)), "zurich", "en", "", api_name="/analyze")
        result = job.result(timeout=args.timeout)
        check_response(result)
        if "Please confirm the material" in result[0]:
            confirmation = result[1]
            selected = confirmation["value"]
            print("Resuming uncertain classification with", selected, flush=True)
            result = client.submit(selected, api_name="/confirm").result(timeout=args.timeout)
            check_response(result)
        print(f"Analyze/confirm elapsed: {time.perf_counter() - started:.2f}s", flush=True)

        result = client.submit(
            handle_file(str(image_path)), "gurugram", "hi", "", "Glass", api_name="/override"
        ).result(timeout=args.timeout)
        check_response(result, expected_rule="gurugram-glass")
        explanation = result[0].split("### Why it helps", 1)[1].split("**Official source:", 1)[0]
        if not any("\u0900" <= character <= "\u097f" for character in explanation):
            raise AssertionError("The Hindi explanation must use Devanagari script")
        result = client.submit(
            handle_file(str(image_path)), "zurich", "de", "", "Glass", api_name="/override"
        ).result(timeout=args.timeout)
        check_response(result, expected_rule="zurich-glass")
    print("PASS: deployed analyze, confirmation when needed, and Hindi/German city overrides.", flush=True)


if __name__ == "__main__":
    main()
