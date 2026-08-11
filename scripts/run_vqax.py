from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from blip_attribution import blur_baseline, integrated_gradients, path_subset, rise, smoothgrad_ig
from panel_utils import PANEL_SIZE, colorize, save_jpeg


METHODS = ("gradcam", "attention", "ig", "smoothgrad_ig", "rise", "pmesa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--indices", default="19506,19507,19508,19510,19512,19513,19514,19511")
    parser.add_argument("--output", type=Path, default=Path("output/qualitative/vqax"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.indices.split(",")]
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    raw: dict[str, dict[str, np.ndarray]] = {}
    manifest = []
    originals: dict[str, Image.Image] = {}

    for position, index in enumerate(indices, 1):
        record = records[str(index)]
        example_id = f"vqax_test_{index}"
        original = Image.open(args.image_dir / Path(record["img_path"]).name).convert("RGB")
        inputs = processor(original, record["question"], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=8)
        answer = processor.decode(generated[0], skip_special_tokens=True)
        labels = processor(text=answer, return_tensors="pt").input_ids.to(args.device)
        pixels = inputs["pixel_values"]
        baseline = blur_baseline(pixels)

        def score(value: torch.Tensor) -> torch.Tensor:
            return -model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=value,
                labels=labels,
            ).loss

        point = pixels.detach().requires_grad_(True)
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=point,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden = output.hidden_states[-1]
        gradient = torch.autograd.grad(-output.loss, hidden)[0]
        gradcam = torch.relu((gradient * hidden).sum(-1)[0, 1:]).reshape(24, 24)
        attention = output.attentions[-1][0, :, 0, 1:].mean(0).reshape(24, 24)
        gradcam = F.interpolate(gradcam[None, None], pixels.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        attention = F.interpolate(attention[None, None], pixels.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        gradcam = gradcam / gradcam.max().clamp_min(1e-12)
        attention = attention / attention.max().clamp_min(1e-12)
        dense = 0.75 * gradcam + 0.25 * attention
        pmesa, selected, completeness = path_subset(score, pixels, baseline, dense, cells=6, paths=2, budget=6)
        maps = {
            "gradcam": gradcam,
            "attention": attention,
            "ig": integrated_gradients(score, pixels, baseline),
            "smoothgrad_ig": smoothgrad_ig(score, pixels, baseline),
            "rise": rise(score, pixels, baseline, samples=48),
            "pmesa": pmesa,
        }
        raw[example_id] = {name: value.detach().cpu().numpy() for name, value in maps.items()}
        originals[example_id] = original
        ground_truth = record["answer"] if isinstance(record["answer"], list) else [record["answer"]]
        with torch.no_grad():
            original_score = float(score(pixels).item())
            baseline_score = float(score(baseline).item())
        manifest.append({
            "example_id": example_id,
            "image": Path(record["img_path"]).name,
            "question": record["question"],
            "ground_truth": ground_truth,
            "predicted_answer": answer,
            "prediction_correct": answer.lower() in {str(value).lower() for value in ground_truth},
            "target": "predicted_answer_log_probability",
            "score": original_score,
            "baseline_score": baseline_score,
            "selected_regions": selected,
            "completeness_error": completeness,
        })
        print(f"[{position}/{len(indices)}] {example_id}: {answer}")

    scales = {
        method: float(np.percentile(np.concatenate([raw[key][method].ravel() for key in raw]), 99))
        for method in METHODS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        example_id = row["example_id"]
        directory = args.output / example_id
        directory.mkdir(parents=True, exist_ok=True)
        save_jpeg(originals[example_id], directory / "input.jpg")
        np.savez_compressed(directory / "raw_maps.npz", **raw[example_id])
        for method in METHODS:
            panel = colorize(originals[example_id], raw[example_id][method], scales[method])
            save_jpeg(panel, directory / f"{method}.jpg")
    report = {
        "dataset": "VQA-X",
        "model": "Salesforce/blip-vqa-base",
        "panel_size": list(PANEL_SIZE),
        "attribution_settings": {"ig_steps": 32, "smoothgrad_ig_samples": 8, "smoothgrad_ig_steps": 16, "smoothgrad_ig_noise_fraction": 0.1, "rise_masks": 48},
        "normalization": {"rule": "global per-method p99", "scales": scales},
        "examples": manifest,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
