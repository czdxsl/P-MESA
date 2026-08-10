"""Run CLIP-based qualitative attribution experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


METHODS = ("saliency", "ig", "smoothgrad_ig", "rise", "kernelshap", "pmesa")


@dataclass
class Example:
    example_id: str
    image: Path
    target_text: str
    source_index: int
    label: str


def mhal_examples(annotation: Path, image_dir: Path, indices: list[int]) -> list[Example]:
    records = json.loads(annotation.read_text(encoding="utf-8"))
    examples: list[Example] = []
    for index in indices:
        record = records[index]
        inaccurate = [a["text"].strip(" ,.") for a in record["annotations"] if a["label"] == "INACCURATE"]
        target = "; ".join(text for text in inaccurate if text)
        if not target:
            raise ValueError(f"M-HalDetect record {index} has no INACCURATE span")
        image = image_dir / record["image"]
        if not image.exists():
            raise FileNotFoundError(image)
        examples.append(Example(f"mhal_val_{index:04d}", image, target, index, "INACCURATE"))
    return examples


def vqax_examples(annotation: Path, image_dir: Path, indices: list[int]) -> list[Example]:
    records = json.loads(annotation.read_text(encoding="utf-8"))
    examples: list[Example] = []
    for index in indices:
        record = records[str(index)]
        if record["dataset"] != "vqaX":
            raise ValueError(f"combined record {index} belongs to {record['dataset']}, not vqaX")
        answer = record["answer"][0] if isinstance(record["answer"], list) else record["answer"]
        rationale = record["explanation"][0]
        target = f"Question: {record['question']} Answer: {answer}. Rationale: {rationale}"
        image = image_dir / Path(record["img_path"]).name
        if not image.exists():
            raise FileNotFoundError(image)
        examples.append(Example(f"vqax_test_{index}", image, target, index, "ANSWER_RATIONALE"))
    return examples


def grid_masks(size: int = 224, cells: int = 7, device: str = "cuda") -> torch.Tensor:
    masks = torch.zeros(cells * cells, size, size, device=device)
    edges = torch.linspace(0, size, cells + 1).round().long()
    k = 0
    for row in range(cells):
        for col in range(cells):
            masks[k, edges[row]:edges[row + 1], edges[col]:edges[col + 1]] = 1
            k += 1
    return masks


def blur_baseline(x: torch.Tensor) -> torch.Tensor:
    low = F.interpolate(x, size=(14, 14), mode="area")
    return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ClipScore:
    def __init__(self, model: CLIPModel, processor: CLIPProcessor, text: str, device: str):
        self.model, self.device = model, device
        tokens = processor(text=[text], return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            feature = model.get_text_features(**tokens)
            self.text = F.normalize(feature, dim=-1)

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        visual = F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
        return (visual * self.text).sum(dim=-1)


def integrated_gradients(score: ClipScore, x: torch.Tensor, baseline: torch.Tensor, steps: int = 16) -> torch.Tensor:
    total = torch.zeros_like(x)
    for alpha in torch.linspace(1 / steps, 1, steps, device=x.device):
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        grad = torch.autograd.grad(score(point).sum(), point)[0]
        total += grad
    return ((x - baseline) * total / steps).abs().sum(dim=1)[0]


def saliency(score: ClipScore, x: torch.Tensor) -> torch.Tensor:
    point = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(score(point).sum(), point)[0]
    return grad.abs().sum(dim=1)[0]


def smoothgrad_ig(score: ClipScore, x: torch.Tensor, baseline: torch.Tensor, samples: int = 4) -> torch.Tensor:
    maps = []
    generator = torch.Generator(device=x.device).manual_seed(1701)
    for _ in range(samples):
        noise = torch.randn(x.shape, generator=generator, device=x.device) * 0.04
        maps.append(integrated_gradients(score, x + noise, baseline, steps=8))
    return torch.stack(maps).mean(0)


def masked_batch(x: torch.Tensor, baseline: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    alpha = F.interpolate(masks[:, None], size=x.shape[-2:], mode="nearest")
    return baseline + alpha * (x - baseline)


@torch.no_grad()
def rise(score: ClipScore, x: torch.Tensor, baseline: torch.Tensor, samples: int = 256) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(2027)
    coarse = (torch.rand(samples, 1, 7, 7, generator=generator, device=x.device) > 0.5).float()
    masks = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    values = []
    for chunk in masks.split(32):
        values.append(score(masked_batch(x, baseline, chunk)))
    weights = torch.cat(values)
    weights = weights - weights.mean()
    return (weights[:, None, None] * masks).mean(0).abs()


@torch.no_grad()
def kernelshap(score: ClipScore, x: torch.Tensor, baseline: torch.Tensor, patches: torch.Tensor, samples: int = 256) -> torch.Tensor:
    n = patches.shape[0]
    generator = torch.Generator(device=x.device).manual_seed(31415)
    z = (torch.rand(samples, n, generator=generator, device=x.device) > 0.5).float()
    z[0], z[1] = 0, 1
    pixel_masks = torch.einsum("sn,nhw->shw", z, patches).clamp(0, 1)
    y = torch.cat([score(masked_batch(x, baseline, chunk)) for chunk in pixel_masks.split(32)])
    design = torch.cat([torch.ones(samples, 1, device=x.device), z], dim=1)
    ridge = torch.eye(n + 1, device=x.device) * 1e-3
    ridge[0, 0] = 0
    coef = torch.linalg.solve(design.T @ design + ridge, design.T @ y)[1:]
    return torch.einsum("n,nhw->hw", coef.abs(), patches)


def pmesa_path_ig(score: ClipScore, x: torch.Tensor, baseline: torch.Tensor, patches: torch.Tensor, paths: int = 4) -> tuple[torch.Tensor, list[int], float]:
    n = patches.shape[0]
    total = torch.zeros(n, device=x.device)
    completeness = []
    rng = random.Random(4242)
    with torch.no_grad():
        delta = float((score(x) - score(baseline)).item())
    for _ in range(paths):
        order = list(range(n)); rng.shuffle(order)
        gates = torch.zeros(n, device=x.device)
        contrib = torch.zeros(n, device=x.device)
        for index in order:
            midpoint = gates.clone(); midpoint[index] = 0.5; midpoint.requires_grad_(True)
            alpha = torch.einsum("n,nhw->hw", midpoint, patches).clamp(0, 1)
            restored = baseline + alpha[None, None] * (x - baseline)
            grad = torch.autograd.grad(score(restored).sum(), midpoint)[0]
            contrib[index] = grad[index]
            gates[index] = 1
        total += contrib
        completeness.append(abs(float(contrib.sum().item()) - delta))
    values = total / paths
    selected = torch.topk(values.abs(), k=min(6, n)).indices.tolist()
    sparse_values = torch.zeros_like(values)
    sparse_values[selected] = values[selected].abs()
    heat = torch.einsum("n,nhw->hw", sparse_values, patches)
    return heat, selected, float(np.mean(completeness))


def colorize(image: Image.Image, heat: np.ndarray, scale: float) -> Image.Image:
    value = np.clip(heat / max(scale, 1e-12), 0, 1)
    rgb = np.stack([np.clip(2.2 * value, 0, 1), np.clip(2.2 * value - 0.7, 0, 1), np.clip(2 * value - 1.5, 0, 1)], -1)
    layer = Image.fromarray(np.uint8(rgb * 255)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), layer, 0.45)


def add_caption(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 42), "white")
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), title, fill="black", font=ImageFont.load_default())
    draw.text((5, 21), subtitle[:90], fill="#4b5563", font=ImageFont.load_default())
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mhaldetect", "vqax"), default="mhaldetect")
    parser.add_argument("--indices", default="4,12,23,31,57,79,83,0")
    parser.add_argument("--output", type=Path, default=Path("results/mhaldetect"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    indices = [int(x) for x in args.indices.split(",")]
    examples = (mhal_examples if args.dataset == "mhaldetect" else vqax_examples)(
        args.annotations, args.image_dir, indices
    )

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patches = grid_masks(device=args.device)
    raw: dict[str, dict[str, np.ndarray]] = {}
    manifest = []

    for position, example in enumerate(examples, 1):
        original = Image.open(example.image).convert("RGB")
        x = processor(images=original, return_tensors="pt")["pixel_values"].to(args.device)
        baseline = blur_baseline(x)
        scorer = ClipScore(model, processor, example.target_text, args.device)
        with torch.no_grad():
            support = float(scorer(x).item())
            baseline_support = float(scorer(baseline).item())
        maps = {
            "saliency": saliency(scorer, x),
            "ig": integrated_gradients(scorer, x, baseline),
            "smoothgrad_ig": smoothgrad_ig(scorer, x, baseline),
            "rise": rise(scorer, x, baseline),
            "kernelshap": kernelshap(scorer, x, baseline, patches),
        }
        pmesa_map, selected, completeness_error = pmesa_path_ig(scorer, x, baseline, patches)
        maps["pmesa"] = pmesa_map
        with torch.no_grad():
            selected_mask = patches[selected].sum(0).clamp(0, 1)
            selected_support = float(scorer(masked_batch(x, baseline, selected_mask[None])).item())
        full_delta = support - baseline_support
        selected_delta = selected_support - baseline_support
        raw[example.example_id] = {name: value.detach().cpu().numpy() for name, value in maps.items()}
        manifest.append({
            "example_id": example.example_id,
            "source_index": example.source_index,
            "source_image": example.image.name,
            "target_span": example.target_text,
            "label": example.label,
            "model": model_name,
            "target_scalar": "CLIP cosine image-text support",
            "support_original": support,
            "support_blur_baseline": baseline_support,
            "support_selected_subset": selected_support,
            "selected_subset_delta_fraction": abs(selected_delta) / max(abs(full_delta), 1e-8),
            "pmesa_selected_patch_indices": selected,
            "pmesa_mean_midpoint_completeness_error": completeness_error,
        })
        print(f"[{position}/{len(examples)}] {example.example_id}: support={support:.4f}")

    scales = {name: float(np.percentile(np.concatenate([raw[e.example_id][name].ravel() for e in examples]), 99)) for name in METHODS}
    args.output.mkdir(parents=True, exist_ok=True)
    for example in examples:
        example_dir = args.output / example.example_id
        example_dir.mkdir(parents=True, exist_ok=True)
        original = Image.open(example.image).convert("RGB")
        add_caption(original, example.example_id, example.target_text).save(example_dir / "input.png")
        np.savez_compressed(example_dir / "raw_maps.npz", **raw[example.example_id])
        for method in METHODS:
            panel = colorize(original, raw[example.example_id][method], scales[method])
            add_caption(panel, method, example.target_text).save(example_dir / f"{method}.png")
    report = {
        "status": "clip_qualitative_experiment",
        "dataset": "M-HalDetect val_raw.json" if args.dataset == "mhaldetect" else "VQA-X test via Uni-NLX combined release",
        "selection": "pre-screened official INACCURATE spans; all requested candidates retained",
        "normalization": {"rule": "global per-method 99th percentile across candidates", "scales": scales},
        "examples": manifest,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
