from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="For example: username/binwise-realwaste-mobilenet")
    parser.add_argument("--folder", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=args.folder, repo_type="model")
    print(f"Uploaded measured artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

