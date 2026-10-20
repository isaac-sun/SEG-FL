"""DeepLabV3 with a MobileNetV3-Large backbone (torchvision), 21 classes.

The backbone starts from the ImageNet weights file given in the config (the
same file torchvision downloads as ``MobileNet_V3_Large_Weights.IMAGENET1K_V1``);
it is shipped with the project because the GPU server may not reach
download.pytorch.org.  The segmentation head is randomly initialised.

Federated state = every parameter plus the BatchNorm running statistics;
``num_batches_tracked`` counters are local only (BN uses a fixed momentum).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger("seg-fl")


class SegModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        self.net = deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)

    def forward(self, x):
        return self.net(x)["out"]

    # ── federated state helpers ───────────────────────────────────────────
    def fl_keys(self) -> list[str]:
        return [k for k in self.state_dict() if not k.endswith("num_batches_tracked")]

    def param_keys(self) -> list[str]:
        return [k for k, _ in self.named_parameters()]

    def get_fl_state(self) -> OrderedDict:
        state = self.state_dict()
        return OrderedDict((k, state[k].detach().clone().cpu()) for k in self.fl_keys())

    def load_fl_state(self, fl_state: OrderedDict) -> None:
        missing, unexpected = self.load_state_dict(fl_state, strict=False)
        bad = [k for k in missing if not k.endswith("num_batches_tracked")]
        if bad or unexpected:
            raise KeyError(f"state mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")


def load_backbone(model: SegModel, path: str) -> int:
    """Copy ImageNet MobileNetV3-Large ``features.*`` weights into the backbone."""
    source = torch.load(path, map_location="cpu", weights_only=True)
    target = model.net.backbone.state_dict()
    loaded = OrderedDict()
    for key, value in source.items():
        if not key.startswith("features."):
            continue
        name = key[len("features."):]
        if name in target and target[name].shape == value.shape:
            loaded[name] = value
    expected = len([k for k in target if not k.endswith("num_batches_tracked")])
    if len(loaded) < expected:
        raise ValueError(f"backbone weights cover {len(loaded)}/{expected} tensors: {path}")
    model.net.backbone.load_state_dict(loaded, strict=False)
    return len(loaded)


def init_head_prior(model: SegModel, class_share) -> None:
    """Set the final classifier bias to log(class share) (floor 1e-4)."""
    share = torch.as_tensor(class_share, dtype=torch.float32).clamp_min(1e-4)
    share = share / share.sum()
    layer = model.net.classifier[-1]
    with torch.no_grad():
        layer.bias.copy_(share.log())


def create_model(config, class_share=None) -> SegModel:
    model = SegModel(config.num_classes)
    path = config.pretrained_backbone
    if path and os.path.exists(path):
        count = load_backbone(model, path)
        logger.info("loaded ImageNet backbone (%d tensors) from %s", count, path)
    elif config.require_pretrained:
        raise FileNotFoundError(
            f"pretrained backbone not found: {path} (use --no-pretrained only for tests)")
    else:
        logger.warning("no pretrained backbone: training from random initialisation")
    if getattr(config, "head_prior_init", False) and class_share is not None:
        init_head_prior(model, class_share)
    return model
