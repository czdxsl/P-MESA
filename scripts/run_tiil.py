from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from blip_attribution import blur_baseline, integrated_gradients, path_subset, rise, smoothgrad_ig
from panel_utils import PANEL_SIZE, colorize, save_jpeg


METHODS = ("patch_similarity", "gradcam", "ig", "smoothgrad_ig", "rise", "pmesa")


def changed_phrases(consistent: str, inconsistent: str) -> tuple[str, str]:
    before, after = consistent.split(), inconsistent.split()
    matcher = difflib.SequenceMatcher(a=before, b=after)
    removed, inserted = [], []
    for operation, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if operation in ("replace", "delete"):
            removed.extend(before[first_start:first_end])
        if operation in ("replace", "insert"):
            inserted.extend(after[second_start:second_end])
    return " ".join(removed).strip(), " ".join(inserted).strip()


def annotation_mask(annotation: dict, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in annotation.get("segmentation", []):
        points = list(zip(polygon[0::2], polygon[1::2]))
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return mask


def ground_truth(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    active = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST), dtype=np.float32) / 255
    green = np.zeros_like(base)
    green[..., 1] = 255
    mixed = base * (1 - 0.35 * active[..., None]) + green * (0.35 * active[..., None])
    return Image.fromarray(np.uint8(mixed.clip(0, 255)))


def localization(heat: np.ndarray, mask: np.ndarray) -> dict[str, float | bool]:
    heat = np.maximum(heat, 0)
    energy = float((heat * mask).sum() / max(heat.sum(), 1e-12))
    point = np.unravel_index(int(heat.argmax()), heat.shape)
    return {"energy_in_mask": energy, "pointing_game": bool(mask[point] > 0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistent", type=Path, required=True)
    parser.add_argument("--inconsistent", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--indices", default="5799,5551,6630,1268,4024,5511,3395,1734,5686,3520,3674,3887,3922,5905,499,944,4138,2005,343,2209")
    parser.add_argument("--output", type=Path, default=Path("output/qualitative/tiil_candidates"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    consistent = json.loads(args.consistent.read_text(encoding="utf-8"))
    inconsistent = json.loads(args.inconsistent.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in consistent["images"]}
    true_annotations = {row["image_id"]: row for row in consistent["annotations"]}
    false_annotations = {row["image_id"]: row for row in inconsistent["annotations"]}
    indices = [int(value) for value in args.indices.split(",")]

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    raw: dict[str, dict[str, np.ndarray]] = {}
    originals: dict[str, Image.Image] = {}
    masks: dict[str, Image.Image] = {}
    manifest = []
    for position, image_id in enumerate(indices, 1):
        image_row = images[image_id]
        true_row, false_row = true_annotations[image_id], false_annotations[image_id]
        original = Image.open(args.data_root / image_row["file_name"]).convert("RGB")
        model_image = original.resize((224, 224), Image.Resampling.BICUBIC)
        pixel_values = processor(images=model_image, return_tensors="pt").pixel_values.to(args.device)
        true_phrase, false_phrase = changed_phrases(true_row["caption"], false_row["caption"])
        true_phrase = true_phrase or true_row["caption"]
        false_phrase = false_phrase or false_row["caption"]
        tokens = processor(
            text=[f"a photo of {true_phrase}", f"a photo of {false_phrase}"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        with torch.no_grad():
            text = model.get_text_features(**tokens)
            text = F.normalize(text, dim=-1)
            direction = text[0] - text[1]
        baseline = blur_baseline(pixel_values)

        def score(value: torch.Tensor) -> torch.Tensor:
            visual = F.normalize(model.get_image_features(pixel_values=value), dim=-1)[0]
            return model.logit_scale.exp() * (visual * direction).sum()

        point = pixel_values.detach().requires_grad_(True)
        vision = model.vision_model(pixel_values=point, output_hidden_states=True, return_dict=True)
        visual = F.normalize(model.visual_projection(vision.pooler_output), dim=-1)[0]
        margin = model.logit_scale.exp() * (visual * direction).sum()
        hidden = vision.hidden_states[-2]
        gradient = torch.autograd.grad(margin, hidden, retain_graph=True)[0]
        gradcam = (gradient * hidden).sum(-1)[0, 1:].abs().reshape(7, 7)
        patches = F.normalize(model.visual_projection(vision.last_hidden_state[0, 1:]), dim=-1)
        patch_similarity = (patches * direction).sum(-1).abs().reshape(7, 7)
        gradcam = F.interpolate(gradcam[None, None], pixel_values.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        patch_similarity = F.interpolate(patch_similarity[None, None], pixel_values.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        gradcam = gradcam / gradcam.max().clamp_min(1e-12)
        patch_similarity = patch_similarity / patch_similarity.max().clamp_min(1e-12)
        dense = 0.75 * gradcam + 0.25 * patch_similarity
        pmesa, selected, completeness = path_subset(score, pixel_values, baseline, dense, cells=6, paths=2, budget=6)
        maps = {
            "patch_similarity": patch_similarity,
            "gradcam": gradcam,
            "ig": integrated_gradients(score, pixel_values, baseline),
            "smoothgrad_ig": smoothgrad_ig(score, pixel_values, baseline),
            "rise": rise(score, pixel_values, baseline, samples=48),
            "pmesa": pmesa,
        }
        example_id = f"tiil_{image_id:05d}"
        raw[example_id] = {name: value.detach().cpu().numpy() for name, value in maps.items()}
        originals[example_id] = original
        mask = annotation_mask(false_row, original.size)
        masks[example_id] = mask
        model_mask = np.asarray(mask.resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32) / 255
        with torch.no_grad():
            original_margin = float(score(pixel_values).item())
            baseline_margin = float(score(baseline).item())
        manifest.append({
            "example_id": example_id,
            "image_id": image_id,
            "image": image_row["file_name"],
            "consistent_caption": true_row["caption"],
            "inconsistent_caption": false_row["caption"],
            "original_phrase": true_phrase,
            "inconsistent_phrase": false_phrase,
            "consistency_margin": original_margin,
            "baseline_margin": baseline_margin,
            "selected_regions": selected,
            "completeness_error": completeness,
            "localization": {name: localization(value.detach().cpu().numpy(), model_mask) for name, value in maps.items()},
        })
        print(f"[{position}/{len(indices)}] {example_id}: margin={original_margin:.3f} energy={manifest[-1]['localization']['pmesa']['energy_in_mask']:.3f}")

    scales = {
        method: float(np.percentile(np.concatenate([raw[key][method].ravel() for key in raw]), 99))
        for method in METHODS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        example_id = row["example_id"]
        directory = args.output / example_id
        save_jpeg(originals[example_id], directory / "input.jpg")
        save_jpeg(ground_truth(originals[example_id], masks[example_id]), directory / "ground_truth.jpg")
        np.savez_compressed(directory / "raw_maps.npz", **raw[example_id])
        for method in METHODS:
            save_jpeg(colorize(originals[example_id], raw[example_id][method], scales[method]), directory / f"{method}.jpg")
    report = {
        "dataset": "TIIL",
        "model": "openai/clip-vit-base-patch32",
        "target": "original_minus_falsified_phrase_similarity",
        "panel_size": list(PANEL_SIZE),
        "attribution_settings": {"ig_steps": 32, "smoothgrad_ig_samples": 8, "smoothgrad_ig_steps": 16, "smoothgrad_ig_noise_fraction": 0.1, "rise_masks": 48},
        "normalization": {"rule": "global per-method p99", "scales": scales},
        "examples": manifest,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
