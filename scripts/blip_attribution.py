from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def blur_baseline(x: torch.Tensor, size: int = 18) -> torch.Tensor:
    low = F.interpolate(x, size=(size, size), mode="area")
    return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)


def grid_masks(size: int, cells: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros(cells * cells, size, size, device=device)
    edges = torch.linspace(0, size, cells + 1, device=device).round().long()
    index = 0
    for row in range(cells):
        for col in range(cells):
            masks[index, edges[row]:edges[row + 1], edges[col]:edges[col + 1]] = 1
            index += 1
    return masks


def restore(x: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return baseline + mask * (x - baseline)


def integrated_gradients(score, x: torch.Tensor, baseline: torch.Tensor, steps: int = 12) -> torch.Tensor:
    total = torch.zeros_like(x)
    for alpha in torch.linspace(1 / steps, 1, steps, device=x.device):
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        total += torch.autograd.grad(score(point), point)[0]
    heat = ((x - baseline) * total / steps).abs().sum(1)
    return F.avg_pool2d(heat[:, None], 15, stride=1, padding=7)[0, 0]


def smoothgrad(score, x: torch.Tensor, samples: int = 8, sigma: float = 0.035) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(1701)
    maps = []
    for _ in range(samples):
        point = (x + torch.randn(x.shape, generator=generator, device=x.device) * sigma).detach().requires_grad_(True)
        maps.append(torch.autograd.grad(score(point), point)[0].abs().sum(1)[0])
    heat = torch.stack(maps).mean(0)
    return F.avg_pool2d(heat[None, None], 15, stride=1, padding=7)[0, 0]


def rise(score, x: torch.Tensor, baseline: torch.Tensor, samples: int = 96, cells: int = 8) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(2027)
    masks = (torch.rand(samples, 1, cells, cells, generator=generator, device=x.device) > 0.5).float()
    masks = F.interpolate(masks, size=x.shape[-2:], mode="bilinear", align_corners=False)
    values = []
    with torch.no_grad():
        base_score = float(score(baseline).item())
        for mask in masks:
            values.append(float(score(restore(x, baseline, mask[None])).item()) - base_score)
    weights = torch.tensor(values, device=x.device)
    weights = weights - weights.mean()
    heat = (weights[:, None, None, None] * masks).mean(0).abs()
    return F.avg_pool2d(heat, 15, stride=1, padding=7)[0]


def path_subset(score, x: torch.Tensor, baseline: torch.Tensor, dense: torch.Tensor, cells: int = 8, paths: int = 3, budget: int = 6) -> tuple[torch.Tensor, list[int], float]:
    masks = grid_masks(x.shape[-1], cells, x.device)
    count = masks.shape[0]
    values = torch.zeros(count, device=x.device)
    errors = []
    rng = random.Random(4242)
    with torch.no_grad():
        delta = float((score(x) - score(baseline)).item())
    for _ in range(paths):
        order = list(range(count))
        rng.shuffle(order)
        gates = torch.zeros(count, device=x.device)
        contribution = torch.zeros(count, device=x.device)
        with torch.no_grad():
            previous = score(baseline)
        for index in order:
            gates[index] = 1
            alpha = torch.einsum("n,nhw->hw", gates, masks).clamp(0, 1)
            with torch.no_grad():
                current = score(restore(x, baseline, alpha[None, None]))
            contribution[index] = current - previous
            previous = current
        values += contribution
        errors.append(abs(float(contribution.sum().item()) - delta))
    values /= paths
    saliency = torch.stack([(dense * mask).mean() for mask in masks])
    saliency = saliency / saliency.max().clamp_min(1e-12)
    positive = values.clamp_min(0)
    contribution = positive if positive.max() > 0 else values.abs()
    ranking = contribution * (0.7 + 0.3 * saliency)
    selected = torch.topk(ranking, min(budget, count)).indices.tolist()
    selection = masks[selected].sum(0).clamp(0, 1)
    selection = F.avg_pool2d(selection[None, None], 31, stride=1, padding=15)[0, 0]
    selection = selection / selection.max().clamp_min(1e-12)
    heat = selection * dense / dense.max().clamp_min(1e-12)
    return heat, selected, float(np.mean(errors))
