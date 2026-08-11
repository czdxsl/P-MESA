from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from pmesa.models import BlipSpanDetector, SpanHead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
    backbone = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", local_files_only=True
    ).to(args.device).eval()
    saved = torch.load(args.checkpoint, map_location="cpu")
    head = SpanHead(backbone.config.text_config.hidden_size)
    head.load_state_dict(saved["head"])
    detector = BlipSpanDetector(backbone, head.to(args.device).eval())
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    rows = []
    for index, record in enumerate(records):
        spans = [
            item["text"].strip(" ,.")
            for item in record["annotations"]
            if item["label"] == "INACCURATE" and item["text"].strip(" ,.")
        ]
        if not spans:
            continue
        span = max(spans, key=len)
        image_path = args.image_dir / record["image"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        pixels = processor(images=image, return_tensors="pt").pixel_values.to(args.device)
        tokens = processor.tokenizer(span, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logit, _ = detector(
                pixels,
                tokens.input_ids.to(args.device),
                tokens.attention_mask.to(args.device),
            )
        rows.append({
            "index": index,
            "probability": float(torch.sigmoid(logit.sum()).item()),
            "landscape": image.width > image.height,
            "span": span,
        })

    rows.sort(key=lambda row: (-row["probability"], row["index"]))
    landscape = [row for row in rows if row["landscape"]]
    print(json.dumps(landscape[:args.top], indent=2))


if __name__ == "__main__":
    main()
