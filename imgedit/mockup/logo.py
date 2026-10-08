"""Logo loading and print preparation (vector-aware)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from ..pixels import parse_color


@dataclass
class Logo:
    path: str
    rgba: np.ndarray  # float32 straight alpha, sRGB 0..1
    vector: bool
    notes: list[str] = field(default_factory=list)
    svg: str = ""  # source markup for vector logos (used for exact per-layer analysis)

    @property
    def aspect(self) -> float:
        return self.rgba.shape[1] / self.rgba.shape[0]


def _svg_aspect(svg: str) -> float:
    """width / height from the viewBox (1.0 if unknown)."""
    import re
    m = re.search(r'viewBox="\s*[-\d.eE]+[\s,]+[-\d.eE]+[\s,]+([\d.eE]+)[\s,]+([\d.eE]+)', svg)
    return float(m.group(1)) / float(m.group(2)) if m else 1.0


def _render_kw(svg: str, long_side: int) -> dict:
    """resvg size argument so the LONG side is long_side (tall artwork must not explode in height)."""
    return {"width": int(long_side)} if _svg_aspect(svg) >= 1 else {"height": int(long_side)}


def _render_svg(path: str, width: int) -> np.ndarray:
    try:
        import resvg_py
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("SVG logos need `pip install resvg-py` (or supply a PNG)") from e
    png = bytes(resvg_py.svg_to_bytes(svg_path=path, **_render_kw(open(path, encoding="utf-8").read(), width)))
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), np.float32) / 255.0


def load_logo(path: str, target_width: int = 2400) -> Logo:
    """Vectors are rasterised large enough that the final warp only ever downsamples."""
    if path.lower().endswith((".svg", ".svgz")):
        rgba = _render_svg(path, target_width)
        logo = Logo(path, rgba, True, svg=open(path, encoding="utf-8").read())
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


def _ink_layers(rgba: np.ndarray, min_share: float = 0.002) -> list[tuple[str, np.ndarray]]:
    """Split artwork into its flat colours (each is one vinyl sheet / one screen).

    Colours are taken only from solid interior pixels (opaque, 3x3 neighbourhood of the same colour),
    so anti-aliased edges between two inks never count as a separate ink.
    """
    vis = rgba[..., 3] > 0.5
    q = np.clip(np.round(rgba[..., :3] * 8), 0, 8).astype(np.int32)
    key = q[..., 0] * 100 + q[..., 1] * 10 + q[..., 2]
    solid = rgba[..., 3] > 0.95
    k = key.astype(np.float32)
    same = (cv2.erode(k, np.ones((3, 3), np.uint8)) == k) & (cv2.dilate(k, np.ones((3, 3), np.uint8)) == k)
    core = solid & same
    vals, cnt = np.unique(key[core], return_counts=True)
    layers = []
    for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1]):
        if c < min_share * max(1, core.sum()):
            continue
        # the layer's full footprint: its own pixels plus edge pixels closest to it
        m = vis & (key == v)
        m = m | (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & vis & ~core)
        r, g, b = (v // 100) / 8, (v // 10 % 10) / 8, (v % 10) / 8
        layers.append(("#%02x%02x%02x" % (int(min(1, r) * 255), int(min(1, g) * 255), int(min(1, b) * 255)), m))
    return layers


def _measure(m: np.ndarray, width_px: int) -> tuple[float, float | None]:
    m8 = m.astype(np.uint8)
    d = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    ridge = (d > 1.0) & (d >= cv2.dilate(d, np.ones((3, 3), np.uint8)) - 1e-6)
    line = float(np.percentile(d[ridge] * 2, 5)) / width_px if ridge.any() else 0.0
    filled = m8.copy()
    cv2.floodFill(filled, np.zeros((m8.shape[0] + 2, m8.shape[1] + 2), np.uint8), (0, 0), 2)
    holes = (filled == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 4)
    dh = cv2.distanceTransform(holes, cv2.DIST_L2, 5)
    widths = [2 * float(dh[lab == k].max()) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 6]
    # enclosed features thinner than ~1/2000 of the artwork width are raster-closed sharp corners
    # (two outline segments almost touching), not holes - a blade cuts them as a corner
    widths = [w for w in widths if w >= max(3.0, width_px / 2000)]
    return line, (min(widths) / width_px if widths else None)


def _svg_layers(svg: str, width_px: int) -> list[tuple[str, np.ndarray]]:
    """Render each <path> alone (others display:none) - exactly what each vinyl sheet / screen gets."""
    import re

    import resvg_py
    tags = list(re.finditer(r"<path\b[^>]*>", svg))
    out = []
    for i, t in enumerate(tags):
        doc = svg
        for j, u in reversed(list(enumerate(tags))):
            if j != i:
                doc = doc[:u.start()] + u.group(0).replace("<path", '<path display="none"', 1) + doc[u.end():]
        png = bytes(resvg_py.svg_to_bytes(svg_string=doc, **_render_kw(svg, width_px)))
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        m = a[..., 3] > 127
        if m.sum() < 20:
            continue
        pid = re.search(r'id="([^"]+)"', t.group(0))
        col = re.search(r'fill="([^"]+)"', t.group(0))
        out.append(((pid.group(1) if pid else f"path{i}") + (f" {col.group(1)}" if col else ""), m))
    return out


def stroke_stats(logo: Logo, work_px: int = 1200) -> list[dict]:
    """Per ink layer: thinnest line and narrowest enclosed gap, as fractions of the logo width.

    Vector logos: each path is rendered on its own (exact; overlapping layers don't fake holes).
    Raster logos: layers are inferred from flat colours. Line = 5th percentile of the medial-axis
    width (ignores tapered tips); gap = smallest inscribed width among a layer's enclosed holes.
    """
    if logo.svg:
        work = 6000
        layers = _svg_layers(logo.svg, work)
        # widths relative to the TRIMMED logo (what the placement's mm refers to)
        alls = np.any([m for _, m in layers], axis=0)
        xs = np.nonzero(alls.any(0))[0]
        width_px = int(xs.max() - xs.min() + 1)  # measurements stay relative to the artwork WIDTH
    else:
        a = logo.rgba
        f = work_px / a.shape[1]
        a = cv2.resize(a, (work_px, max(2, int(round(a.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        layers = _ink_layers(a)
        width_px = work_px
    out = []
    total = max(1, int(np.sum([m.sum() for _, m in layers])))
    for name, m in layers:
        line, gap = _measure(m, width_px)
        out.append({"colour": name, "share": float(m.sum() / total), "line": line, "gap": gap})
    return out
