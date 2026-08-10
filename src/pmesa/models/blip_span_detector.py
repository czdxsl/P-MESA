from __future__ import annotations

import torch
from torch import nn


class SpanHead(nn.Module):
    def __init__(self, hidden_size: int = 768, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class BlipSpanDetector(nn.Module):
    def __init__(self, backbone: nn.Module, head: SpanHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def encode(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vision = self.backbone.vision_model(pixel_values=pixel_values)
        image = vision.last_hidden_state
        image_mask = torch.ones(image.shape[:2], dtype=torch.long, device=image.device)
        text = self.backbone.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image,
            encoder_attention_mask=image_mask,
            return_dict=True,
        )
        return text.last_hidden_state[:, 0], image

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, image = self.encode(pixel_values, input_ids, attention_mask)
        return self.head(features), image
