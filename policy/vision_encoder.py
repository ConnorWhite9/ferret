"""Frozen ResNet-50 target model and input encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from ferret.constants import INPUT_EMBED_DIM, NUM_CLASSES


class VisionEncoder(nn.Module):
    """Frozen ImageNet ResNet-50 returning logits and pooled features."""

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = backbone.fc
        self.to(self.device)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return [B, 2048] feature vectors."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        feats = self.features(image).flatten(1)
        return feats

    @torch.inference_mode()
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        """Return [B, num_classes] logits."""
        return self.logits_with_grad(image).detach()

    def logits_with_grad(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass that preserves gradients w.r.t. the input image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        feats = self.features(image).flatten(1)
        return self.classifier(feats)

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(image)
        logits = self.classifier(feats)
        return logits, feats

    @property
    def feature_dim(self) -> int:
        return INPUT_EMBED_DIM

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES
