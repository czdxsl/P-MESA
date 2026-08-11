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


def gaussian_blur(heat: torch.Tensor, kernel_size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    coordinates = torch.arange(kernel_size, device=heat.device, dtype=heat.dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    value = heat[None, None]
    value = F.conv2d(value, kernel[None, None, :, None], padding=(kernel_size // 2, 0))
    value = F.conv2d(value, kernel[None, None, None, :], padding=(0, kernel_size // 2))
    return value[0, 0]


def integrated_attribution(score, x: torch.Tensor, baseline: torch.Tensor, steps: int) -> torch.Tensor:
    total = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)
    for index, alpha in enumerate(alphas):
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        weight = 0.5 if index in (0, steps) else 1.0
        total += weight * torch.autograd.grad(score(point), point)[0]
    return (x - baseline) * total / steps


def integrated_gradients(score, x: torch.Tensor, baseline: torch.Tensor, steps: int = 32) -> torch.Tensor:
    attribution = integrated_attribution(score, x, baseline, steps)
    heat = attribution.square().sum(1).sqrt()[0]
    return gaussian_blur(heat)


def smoothgrad_ig(score, x: torch.Tensor, baseline: torch.Tensor, samples: int = 8, steps: int = 16, noise_fraction: float = 0.1) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(1701)
    total = torch.zeros_like(x)
    lower, upper = x.min().detach(), x.max().detach()
    noise_scale = (upper - lower).clamp_min(1e-6) * noise_fraction
    for _ in range(samples):
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * noise_scale
        noisy = (x + noise).clamp(lower, upper)
        total += integrated_attribution(score, noisy, baseline, steps)
    attribution = total / samples
    heat = attribution.square().sum(1).sqrt()[0]
    return gaussian_blur(heat)


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
