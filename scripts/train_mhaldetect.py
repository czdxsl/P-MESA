from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import BlipForQuestionAnswering, BlipProcessor

from pmesa.models import SpanHead


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predicted = logits.sigmoid() >= 0.5
    target = labels.bool()
    true_positive = (predicted & target).sum().item()
    false_positive = (predicted & ~target).sum().item()
    false_negative = (~predicted & target).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (predicted == target).float().mean().item()
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def extract(records, names, image_dir, processor, backbone, device):
    by_image = {name: [] for name in names}
    for record in records:
        if record["image"] not in by_image:
            continue
        for annotation in record["annotations"]:
            if annotation["label"] in ("ACCURATE", "INACCURATE") and annotation["text"].strip():
                by_image[record["image"]].append((annotation["text"].strip(), int(annotation["label"] == "INACCURATE")))
    features, labels, groups = [], [], []
    for position, name in enumerate(names, 1):
        pairs = by_image[name]
        if not pairs:
            continue
        image = Image.open(image_dir / name).convert("RGB")
        dtype = next(backbone.parameters()).dtype
        pixels = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=dtype)
        vision = backbone.vision_model(pixel_values=pixels).last_hidden_state
        for start in range(0, len(pairs), 24):
            chunk = pairs[start:start + 24]
            tokens = processor.tokenizer([text for text, _ in chunk], padding=True, truncation=True, max_length=64, return_tensors="pt")
            ids = tokens.input_ids.to(device)
            mask = tokens.attention_mask.to(device)
            image_states = vision.expand(len(chunk), -1, -1)
            image_mask = torch.ones(image_states.shape[:2], dtype=torch.long, device=device)
            encoded = backbone.text_encoder(
                input_ids=ids,
                attention_mask=mask,
                encoder_hidden_states=image_states,
                encoder_attention_mask=image_mask,
                return_dict=True,
            ).last_hidden_state[:, 0]
            features.append(encoded.float().cpu())
            labels.extend(label for _, label in chunk)
            groups.extend([name] * len(chunk))
        print(f"[{position}/{len(names)}] {name} {len(pairs)}")
    return torch.cat(features), torch.tensor(labels, dtype=torch.float32), groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mhaldetect_span_head.pt"))
    parser.add_argument("--feature-cache", type=Path, default=Path("results/mhaldetect/features.pt"))
    parser.add_argument("--max-images", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    available = sorted({record["image"] for record in records if (args.image_dir / record["image"]).exists()})
    random.Random(args.seed).shuffle(available)
    names = available[:args.max_images]
    if len(names) < 2:
        raise SystemExit("at least two downloaded images are required")
    split = max(1, int(len(names) * 0.85))
    train_names, val_names = set(names[:split]), set(names[split:])

    cached = torch.load(args.feature_cache, map_location="cpu") if args.feature_cache.exists() else None
    if cached is not None and cached.get("names") == names:
        features, labels, groups = cached["features"], cached["labels"], cached["groups"]
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
        dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
        backbone = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", local_files_only=True, torch_dtype=dtype
        ).to(args.device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        features, labels, groups = extract(records, names, args.image_dir, processor, backbone, args.device)
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels, "groups": groups, "names": names}, args.feature_cache)

    train_index = torch.tensor([group in train_names for group in groups])
    val_index = torch.tensor([group in val_names for group in groups])
    train_features, train_labels = features[train_index], labels[train_index]
    val_features, val_labels = features[val_index], labels[val_index]
    counts = torch.bincount(train_labels.long(), minlength=2).float()
    weights = (1 / counts.clamp_min(1))[train_labels.long()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=512, sampler=sampler)

    head = SpanHead(features.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for batch_features, batch_labels in loader:
            logits = head(batch_features.to(args.device))
            loss = criterion(logits, batch_labels.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.no_grad():
            logits = head(val_features.to(args.device)).cpu()
        result = metrics(logits, val_labels)
        result.update(epoch=epoch, loss=sum(losses) / len(losses))
        history.append(result)
        print(json.dumps(result))
        if result["f1"] > best:
            best = result["f1"]
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head": best_state,
        "model": "Salesforce/blip-vqa-base",
        "seed": args.seed,
        "train_images": len(train_names),
        "val_images": len(val_names),
        "train_spans": int(train_index.sum()),
        "val_spans": int(val_index.sum()),
        "best_f1": best,
        "history": history,
    }, args.checkpoint)
    print(args.checkpoint.resolve())


if __name__ == "__main__":
    main()
