from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download(name: str, output: Path) -> tuple[str, int]:
    target = output / name
    if target.exists() and target.stat().st_size > 1000:
        return name, target.stat().st_size
    split = "train2014" if "train2014" in name else "val2014"
    response = requests.get(f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{name}", timeout=90)
    response.raise_for_status()
    target.write_bytes(response.content)
    return name, len(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/vqax_images"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.indices.split(",")]
    names = []
    for index in indices:
        name = Path(records[str(index)]["img_path"]).name
        if name not in names:
            names.append(name)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(download, name, args.output) for name in names]
        for position, job in enumerate(as_completed(jobs), 1):
            name, size = job.result()
            print(f"[{position}/{len(names)}] {name} {size}")


if __name__ == "__main__":
    main()
