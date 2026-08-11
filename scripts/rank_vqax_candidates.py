from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from blip_attribution import blur_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.indices.split(",")]
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", local_files_only=True
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    for index in indices:
        record = records[str(index)]
        image = Image.open(args.image_dir / Path(record["img_path"]).name).convert("RGB")
        inputs = processor(image, record["question"], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=8)
        answer = processor.decode(generated[0], skip_special_tokens=True)
        labels = processor(text=answer, return_tensors="pt").input_ids.to(args.device)
        pixels = inputs["pixel_values"]
        baseline = blur_baseline(pixels)

        def score(value: torch.Tensor) -> float:
            with torch.no_grad():
                loss = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=value,
                    labels=labels,
                ).loss
            return float((-loss).item())

        answers = record["answer"] if isinstance(record["answer"], list) else [record["answer"]]
        rows.append({
            "index": index,
            "answer": answer,
            "correct": answer.lower() in {str(value).lower() for value in answers},
            "landscape": image.width > image.height,
            "score_delta": score(pixels) - score(baseline),
        })


    rows.sort(key=lambda row: (-row["score_delta"], row["index"]))
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
