from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download(name: str, output: Path) -> tuple[str, int]:
    target = output / name
    if target.exists() and target.stat().st_size > 1000:
        return name, target.stat().st_size
    response = requests.get(f"https://images.cocodataset.org/val2014/{name}", timeout=90)
    response.raise_for_status()
    target.write_bytes(response.content)
    return name, len(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/mhaldetect/images"))
    parser.add_argument("--max-images", type=int, default=700)
    parser.add_argument("--indices")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if args.indices:
        names = sorted({records[int(index)]["image"] for index in args.indices.split(",")})
    else:
        names = sorted({record["image"] for record in records})
        random.Random(args.seed).shuffle(names)
        names = names[:args.max_images]
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download, name, args.output): name for name in names}
        for position, job in enumerate(as_completed(jobs), 1):
            name, size = job.result()
            print(f"[{position}/{len(names)}] {name} {size}")


if __name__ == "__main__":
    main()
