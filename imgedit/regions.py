"""Region masks. Coordinates are normalized to the canvas at the time the step runs.

A mask is a float32 HxW array in [0, 1]. ``None`` means "whole canvas".
Regions are geometric (rect / ellipse / polygon) optionally intersected with a
color-range selector. They are *not* semantic segmentation, which is a stated
limit of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .pixels import gaussian_blur, luma, rgb_to_hsv

SHAPES = ("full", "rect", "ellipse", "polygon")


@dataclass
class ColorRange:
    hue_center: float  # degrees
    hue_tolerance: float = 25.0  # degrees, half-width
    min_saturation: float = 0.15
    lum_min: float = 0.0
    lum_max: float = 1.0


@dataclass
class Region:
    shape: str = "full"
    box: Optional[list[float]] = None  # [x0, y0, x1, y1] normalized
    points: Optional[list[list[float]]] = None  # [[x, y], ...] normalized
    feather: float = 0.0  # fraction of the canvas short side
    invert: bool = False
    color_range: Optional[ColorRange] = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return self.shape == "full" and self.color_range is None and not self.invert

    def describe(self) -> str:
        if self.shape == "full":
            base = "whole image"
        elif self.shape == "polygon":
            base = f"polygon ({len(self.points or [])} pts)"
        else:
            x0, y0, x1, y1 = self.box or (0, 0, 1, 1)
            base = f"{self.shape} x {x0:.0%}-{x1:.0%}, y {y0:.0%}-{y1:.0%}"
        if self.color_range:
            c = self.color_range
            base += f" ∩ hue {c.hue_center:.0f}°±{c.hue_tolerance:.0f}°"
            if c.lum_min > 0 or c.lum_max < 1:
                base += f", luminance {c.lum_min:.2f}-{c.lum_max:.2f}"
        if self.invert:
            base = f"everything except ({base})"
        if self.feather > 0:
            base += f", feather {self.feather:.1%}"
        return base


def _shape_mask(region: Region, h: int, w: int) -> np.ndarray:
    if region.shape == "full":
        return np.ones((h, w), np.float32)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    if region.shape in ("rect", "ellipse"):
        x0, y0, x1, y1 = region.box  # validated upstream
        xy = [x0 * w, y0 * h, x1 * w - 1, y1 * h - 1]
        xy[2] = max(xy[2], xy[0])
        xy[3] = max(xy[3], xy[1])
        (draw.rectangle if region.shape == "rect" else draw.ellipse)(xy, fill=255)
    elif region.shape == "polygon":
        draw.polygon([(x * w, y * h) for x, y in region.points], fill=255)
    return np.asarray(img, np.float32) / 255.0


def _color_mask(rgb: np.ndarray, cr: ColorRange) -> np.ndarray:
    h, s, _ = rgb_to_hsv(rgb)
    dist = np.abs((h - cr.hue_center + 180.0) % 360.0 - 180.0)
    soft = max(cr.hue_tolerance * 0.35, 1.0)
    hue_w = np.clip((cr.hue_tolerance + soft - dist) / soft, 0, 1)
    sat_w = np.clip((s - cr.min_saturation) / 0.08 + 0.5, 0, 1)
    lum = luma(rgb)
    lum_w = np.clip((lum - cr.lum_min) / 0.04 + 0.5, 0, 1) * np.clip((cr.lum_max - lum) / 0.04 + 0.5, 0, 1)
    return (hue_w * sat_w * lum_w).astype(np.float32)


def build_mask(region: Region, rgb: np.ndarray) -> Optional[np.ndarray]:
    """Return the soft mask for a region, or None when the region is the whole canvas."""
    if region.is_full:
        return None
    h, w = rgb.shape[:2]
    m = _shape_mask(region, h, w)
    if region.color_range is not None:
        m = m * _color_mask(rgb, region.color_range)
    if region.invert:
        m = 1.0 - m
    if region.feather > 0:
        m = gaussian_blur(m, region.feather * min(h, w) / 2.0)
        # Box-blur tails can leave float dust; snap it so "outside" stays exactly 0.
        m[m < 1e-4] = 0.0
        m[m > 1 - 1e-4] = 1.0
    return np.clip(m, 0.0, 1.0).astype(np.float32)
