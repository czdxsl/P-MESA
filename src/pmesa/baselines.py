"""Task-valid low-evidence baseline utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def low_evidence_image(image: torch.Tensor, *, downsample_factor: int = 16, blur_kernel: int = 9) -> torch.Tensor:
    """Downsample, reconstruct, and blur a BCHW or CHW image tensor."""
    squeeze = image.ndim == 3
    x = image.unsqueeze(0) if squeeze else image
    if x.ndim != 4:
        raise ValueError("image must be CHW or BCHW")
    h, w = x.shape[-2:]
    low = F.interpolate(x, size=(max(1, h // downsample_factor), max(1, w // downsample_factor)), mode="area")
    restored = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        coords = torch.arange(blur_kernel, device=x.device, dtype=x.dtype) - blur_kernel // 2
        sigma = max(blur_kernel / 6.0, 0.5)
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        channels = x.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        pad = blur_kernel // 2
        horizontal_mode = "reflect" if pad < restored.shape[-1] else "replicate"
        vertical_mode = "reflect" if pad < restored.shape[-2] else "replicate"
        restored = F.pad(restored, (pad, pad, 0, 0), mode=horizontal_mode)
        restored = F.conv2d(restored, horizontal, groups=channels)
        restored = F.pad(restored, (0, 0, pad, pad), mode=vertical_mode)
        restored = F.conv2d(restored, vertical, groups=channels)
    return restored.squeeze(0) if squeeze else restored


def restore_regions(baseline: torch.Tensor, original: torch.Tensor, masks: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Softly restore possibly-overlapping region masks.

    Alpha union avoids values above one where masks overlap.
    """
    if masks.shape[0] != gates.numel():
        raise ValueError("one restoration gate is required per mask")
    masks = masks.to(device=original.device, dtype=original.dtype)
    gates = gates.to(device=original.device, dtype=original.dtype)
    alpha = 1.0 - torch.prod(1.0 - masks * gates.view(-1, 1, 1), dim=0)
    return baseline * (1.0 - alpha) + original * alpha
