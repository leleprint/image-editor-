"""Logo loading and print preparation (vector-aware)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..pixels import parse_color


@dataclass
class Logo:
    path: str
    rgba: np.ndarray  # float32 straight alpha, sRGB 0..1
    vector: bool
    notes: list[str] = field(default_factory=list)

    @property
    def aspect(self) -> float:
        return self.rgba.shape[1] / self.rgba.shape[0]


def _render_svg(path: str, width: int) -> np.ndarray:
    try:
        import resvg_py
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("SVG logos need `pip install resvg-py` (or supply a PNG)") from e
    png = bytes(resvg_py.svg_to_bytes(svg_path=path, width=int(width)))
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), np.float32) / 255.0


def load_logo(path: str, target_width: int = 2400) -> Logo:
    """Vectors are rasterised large enough that the final warp only ever downsamples."""
    if path.lower().endswith((".svg", ".svgz")):
        rgba = _render_svg(path, target_width)
        logo = Logo(path, rgba, True)
    else:
        im = Image.open(path)
        im.load()
        logo = Logo(path, np.asarray(im.convert("RGBA"), np.float32) / 255.0, False)
        if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
            logo.notes.append("logo has no transparency: its background would be printed too")
    return trim(logo)


def trim(logo: Logo) -> Logo:
    a = logo.rgba[..., 3]
    ys, xs = np.nonzero(a > 0.01)
    if len(xs) == 0:
        raise ValueError("logo is fully transparent")
    pad = 2
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
    logo.rgba = logo.rgba[y0:y1, x0:x1].copy()
    return logo


def has_opaque_light_background(logo: Logo) -> bool:
    a = logo.rgba
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return bool((border[:, 3] > 0.98).mean() > 0.9 and (border[:, :3].min(-1) > 0.92).mean() > 0.9)


def knockout_white(logo: Logo) -> Logo:
    """Remove a white background: alpha from distance to white, colour un-mixed from white."""
    rgb, a = logo.rgba[..., :3], logo.rgba[..., 3]
    na = np.clip((1.0 - rgb.min(-1)) / 0.9, 0, 1)
    na = np.where(na < 0.04, 0, na)
    col = np.where(na[..., None] > 1e-3, 1 - (1 - rgb) / np.maximum(na[..., None], 1e-3), 0)
    logo.rgba = np.dstack([np.clip(col, 0, 1), na * a]).astype(np.float32)
    logo.notes.append("white background knocked out")
    return trim(logo)


def recolor(logo: Logo, color: str) -> Logo:
    c = np.array(parse_color(color), np.float32)
    logo.rgba = np.dstack([np.broadcast_to(c, logo.rgba.shape[:2] + (3,)), logo.rgba[..., 3]]).astype(np.float32)
    logo.notes.append(f"recoloured to single ink {color}")
    return logo


def preview_png(logo: Logo, max_side: int = 256) -> bytes:
    """Logo on mid-grey so white artwork stays visible to the planner."""
    im = Image.fromarray((np.clip(logo.rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    bg.alpha_composite(im)
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def describe(logo: Logo) -> str:
    a = logo.rgba
    vis = a[..., 3] > 0.5
    cols = a[vis][:, :3]
    q = np.round(cols * 4) / 4
    uniq, cnt = np.unique(q, axis=0, return_counts=True)
    top = uniq[np.argsort(-cnt)[:3]]
    hexes = ["#%02x%02x%02x" % tuple((c * 255).astype(int)) for c in top]
    return (f"{os.path.basename(logo.path)}: {'vector' if logo.vector else 'raster'} {a.shape[1]}x{a.shape[0]}, "
            f"aspect {logo.aspect:.2f}, coverage {vis.mean():.0%}, main colours {', '.join(hexes)}")
