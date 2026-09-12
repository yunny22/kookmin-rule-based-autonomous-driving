from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthwiseRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.block(feature)


class FarCenterlineModel(nn.Module):
    """Spatially aligned per-row x-bin/no-line classifier."""

    def __init__(
        self,
        *,
        x_bin_count: int = 128,
        feature_channels: int = 32,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained_backbone
            else None
        )
        features = list(mobilenet_v3_small(weights=weights).features.children())
        self.stage4 = nn.Sequential(*features[:2])
        self.stage8 = nn.Sequential(*features[2:4])
        self.stage16 = nn.Sequential(*features[4:9])
        channels = int(feature_channels)
        self.lateral4 = nn.Conv2d(16, channels, kernel_size=1, bias=False)
        self.lateral8 = nn.Conv2d(24, channels, kernel_size=1, bias=False)
        self.lateral16 = nn.Conv2d(48, channels, kernel_size=1, bias=False)
        self.refine8 = DepthwiseRefine(channels)
        self.refine4 = DepthwiseRefine(channels)
        self.location_head = nn.Sequential(
            DepthwiseRefine(channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.no_line_head = nn.Sequential(
            nn.Conv1d(channels * 2, 16, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1),
        )
        self.x_bin_count = int(x_bin_count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature4 = self.stage4(image)
        feature8 = self.stage8(feature4)
        feature16 = self.stage16(feature8)
        pyramid8 = self.lateral8(feature8) + F.interpolate(
            self.lateral16(feature16),
            size=feature8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid8 = self.refine8(pyramid8)
        pyramid4 = self.lateral4(feature4) + F.interpolate(
            pyramid8,
            size=feature4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid4 = self.refine4(pyramid4)

        location = self.location_head(pyramid4).squeeze(1)
        if location.shape[-1] != self.x_bin_count:
            location = F.interpolate(
                location.unsqueeze(1),
                size=(location.shape[-2], self.x_bin_count),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        row_feature = torch.cat(
            (pyramid4.mean(dim=3), pyramid4.amax(dim=3)), dim=1
        )
        no_line = self.no_line_head(row_feature).transpose(1, 2)
        return torch.cat((location, no_line), dim=2)


def anchor_rows_for_stride(height: int, stride: int = 4) -> tuple[int, ...]:
    count = max(1, int(height) // int(stride))
    return tuple(
        min(int(height) - 1, int(stride) * index + int(stride) // 2)
        for index in range(count)
    )


def row_interpolation_matrix(
    anchor_rows: tuple[int, ...], height: int
) -> np.ndarray:
    anchors = np.asarray(anchor_rows, dtype=np.int64)
    matrix = np.zeros((int(height), len(anchor_rows)), dtype=np.float32)
    for row in range(int(height)):
        if row < anchors[0] or row > anchors[-1]:
            continue
        right = int(np.searchsorted(anchors, row, side="left"))
        if right == 0:
            matrix[row, 0] = 1.0
        elif right >= anchors.size:
            matrix[row, -1] = 1.0
        elif anchors[right] == row:
            matrix[row, right] = 1.0
        else:
            left = right - 1
            weight = float(row - anchors[left]) / float(
                anchors[right] - anchors[left]
            )
            matrix[row, left] = 1.0 - weight
            matrix[row, right] = weight
    return matrix


class FarCenterlineMaskAdapter(nn.Module):
    """Render confident x-bin classifications as a thin yellow-only mask."""

    def __init__(
        self,
        core: nn.Module,
        *,
        output_width: int = 512,
        output_height: int = 288,
        x_bin_count: int = 128,
        line_presence_threshold: float = 0.55,
        conditional_x_threshold: float = 0.10,
        line_half_width_px: float = 2.5,
        line_sharpness: float = 2.0,
    ) -> None:
        super().__init__()
        self.core = core
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.x_bin_count = int(x_bin_count)
        self.line_presence_threshold = float(line_presence_threshold)
        self.conditional_x_threshold = float(conditional_x_threshold)
        self.line_half_width_px = float(line_half_width_px)
        self.line_sharpness = float(line_sharpness)
        anchors = anchor_rows_for_stride(self.output_height)
        self.register_buffer(
            "row_interpolation",
            torch.from_numpy(
                row_interpolation_matrix(anchors, self.output_height)
            ),
        )
        self.register_buffer(
            "x_pixels",
            torch.arange(self.output_width, dtype=torch.float32).reshape(
                1, 1, self.output_width
            ),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.core(image)
        probabilities = torch.softmax(logits, dim=2)
        line_probabilities = probabilities[..., : self.x_bin_count]
        no_line_probability = probabilities[..., self.x_bin_count]
        best_probability, best_bin = line_probabilities.max(dim=2)
        line_presence = 1.0 - no_line_probability
        conditional_probability = best_probability / line_presence.clamp_min(
            1.0e-6
        )
        visible = (
            (line_presence >= self.line_presence_threshold)
            & (conditional_probability >= self.conditional_x_threshold)
        ).to(logits.dtype)
        x = best_bin.to(logits.dtype) / float(self.x_bin_count - 1)

        interpolation = self.row_interpolation.transpose(0, 1)
        curve_x = torch.matmul(x, interpolation) * float(
            self.output_width - 1
        )
        curve_visible = torch.matmul(visible, interpolation)
        distance = torch.abs(self.x_pixels - curve_x.unsqueeze(-1))
        line_logit = self.line_sharpness * (
            self.line_half_width_px - distance
        )
        yellow = torch.where(
            curve_visible.unsqueeze(-1) >= 0.5,
            line_logit,
            torch.full_like(line_logit, -20.0),
        )
        background = torch.zeros_like(yellow)
        white = torch.full_like(yellow, -20.0)
        return torch.stack((background, white, yellow), dim=1)


def expected_x_from_logits(
    logits: torch.Tensor, x_bin_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=2)
    line = probabilities[..., : int(x_bin_count)]
    line_presence = 1.0 - probabilities[..., int(x_bin_count)]
    conditional = line / line.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
    bins = torch.linspace(
        0.0,
        1.0,
        int(x_bin_count),
        device=logits.device,
        dtype=logits.dtype,
    )
    expected = (conditional * bins).sum(dim=2)
    confidence = conditional.amax(dim=2)
    return expected, line_presence, confidence
