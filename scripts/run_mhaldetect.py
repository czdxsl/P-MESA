from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipForQuestionAnswering, BlipProcessor

from blip_attribution import blur_baseline, integrated_gradients, path_subset, rise, smoothgrad
from pmesa.models import BlipSpanDetector, SpanHead


METHODS = ("hallucination", "counterevidence", "ig", "smoothgrad", "rise", "pmesa")


def colorize(image: Image.Image, heat: np.ndarray, scale: float) -> Image.Image:
    value = np.clip(heat / max(scale, 1e-12), 0, 1)
    rgb = np.stack([
        np.clip(2.2 * value, 0, 1),
        np.clip(2.2 * value - 0.7, 0, 1),
        np.clip(2 * value - 1.5, 0, 1),
    ], -1)
    layer = Image.fromarray(np.uint8(rgb * 255)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), layer, 0.45)


def caption(image: Image.Image, title: str, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), title, fill="black", font=ImageFont.load_default())
    draw.text((5, 21), text[:100], fill="#4b5563", font=ImageFont.load_default())
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--indices", default="4,12,23,31,57,79,83,0")
    parser.add_argument("--output", type=Path, default=Path("results/mhaldetect"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.indices.split(",")]
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
    backbone = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True).to(args.device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(args.checkpoint, map_location="cpu")
    head = SpanHead(backbone.config.text_config.hidden_size)
    head.load_state_dict(saved["head"])
    detector = BlipSpanDetector(backbone, head.to(args.device).eval())
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    raw: dict[str, dict[str, np.ndarray]] = {}
    manifest = []
    originals: dict[str, Image.Image] = {}

    for position, index in enumerate(indices, 1):
        record = records[index]
        example_id = f"mhal_val_{index:04d}"
        spans = [annotation["text"].strip(" ,.") for annotation in record["annotations"] if annotation["label"] == "INACCURATE"]
        span = max((text for text in spans if text), key=len)
        original = Image.open(args.image_dir / record["image"]).convert("RGB")
        image_inputs = processor(images=original, return_tensors="pt")
        tokens = processor.tokenizer(span, truncation=True, max_length=64, return_tensors="pt")
        pixels = image_inputs.pixel_values.to(args.device)
        input_ids = tokens.input_ids.to(args.device)
        attention_mask = tokens.attention_mask.to(args.device)
        baseline = blur_baseline(pixels)

        def score(value: torch.Tensor) -> torch.Tensor:
            return detector(value, input_ids, attention_mask)[0].sum()

        point = pixels.detach().requires_grad_(True)
        logit, image_states = detector(point, input_ids, attention_mask)
        gradient = torch.autograd.grad(logit.sum(), image_states)[0]
        signed = (gradient * image_states).sum(-1)[0, 1:].reshape(24, 24)
        hallucination = torch.relu(signed)
        counterevidence = torch.relu(-signed)
        hallucination = F.interpolate(hallucination[None, None], pixels.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        counterevidence = F.interpolate(counterevidence[None, None], pixels.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        dense = torch.relu(signed)
        dense = F.interpolate(dense[None, None], pixels.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        dense = dense / dense.max().clamp_min(1e-12)
        pmesa, selected, completeness = path_subset(score, pixels, baseline, dense, cells=6, paths=2, budget=6)
        maps = {
            "hallucination": hallucination,
            "counterevidence": counterevidence,
            "ig": integrated_gradients(score, pixels, baseline),
            "smoothgrad": smoothgrad(score, pixels),
            "rise": rise(score, pixels, baseline, samples=48),
            "pmesa": pmesa,
        }
        raw[example_id] = {name: value.detach().cpu().numpy() for name, value in maps.items()}
        originals[example_id] = original
        with torch.no_grad():
            original_logit = float(score(pixels).item())
            baseline_logit = float(score(baseline).item())
        manifest.append({
            "example_id": example_id,
            "image": record["image"],
            "span": span,
            "label": "INACCURATE",
            "hallucination_probability": float(torch.sigmoid(torch.tensor(original_logit)).item()),
            "logit": original_logit,
            "baseline_logit": baseline_logit,
            "selected_regions": selected,
            "completeness_error": completeness,
        })
        print(f"[{position}/{len(indices)}] {example_id}: {manifest[-1]['hallucination_probability']:.4f}")

    scales = {
        method: float(np.percentile(np.concatenate([raw[key][method].ravel() for key in raw]), 99))
        for method in METHODS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        example_id = row["example_id"]
        directory = args.output / example_id
        directory.mkdir(parents=True, exist_ok=True)
        text = f"p={row['hallucination_probability']:.2f} | {row['span']}"
        caption(originals[example_id], example_id, text).save(directory / "input.png")
        np.savez_compressed(directory / "raw_maps.npz", **raw[example_id])
        for method in METHODS:
            panel = colorize(originals[example_id], raw[example_id][method], scales[method])
            caption(panel, method, text).save(directory / f"{method}.png")
    report = {
        "dataset": "M-HalDetect",
        "model": "BLIP span detector",
        "checkpoint": str(args.checkpoint),
        "training": {key: saved[key] for key in ("seed", "train_images", "val_images", "train_spans", "val_spans", "best_f1")},
        "normalization": {"rule": "global per-method p99", "scales": scales},
        "examples": manifest,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
