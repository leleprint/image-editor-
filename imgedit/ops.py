"""Operation catalog.

Every operation is deterministic, has a typed parameter spec with hard bounds,
and declares the artifacts it can introduce. The catalog text shown to the
planner model is generated from this file, so the model can only ask for what
the engine can actually do.

Kinds:
  tonal     - per-pixel/neighbourhood change, honours a region mask
  overlay   - draws new content; footprint limited to what it draws
  geometric - changes canvas size/orientation; always whole-canvas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import pixels as px


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # float | int | bool | str | color | enum
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    choices: tuple = ()
    desc: str = ""
    required: bool = False


@dataclass(frozen=True)
class Op:
    name: str
    kind: str
    summary: str
    params: tuple[Param, ...]
    fn: Callable
    risks: tuple[str, ...] = ()
    lossless: bool = False  # no resampling/quantization beyond the final 8-bit encode

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)


@dataclass
class Canvas:
    """Working state. ``layers`` are carried through geometric ops in lockstep."""

    rgb: np.ndarray  # HxWx3 edited image
    alpha: np.ndarray  # HxW
    ref_rgb: np.ndarray  # original, carried through the same geometric ops
    ref_alpha: np.ndarray
    touched: np.ndarray  # HxW union of all edit masks
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


# --------------------------------------------------------------------- tonal

def _exposure(rgb, stops, **_):
    return px.to_srgb(px.to_linear(rgb) * (2.0 ** stops))


def _brightness(rgb, amount, **_):
    # Gamma curve: lifts/lowers midtones while pinning pure black and white.
    return np.clip(rgb, 0, 1) ** (2.0 ** (-amount))


def _smoothstep(x):
    return x * x * (3 - 2 * x)


def _contrast(rgb, amount, **_):
    x = np.clip(rgb, 0, 1)
    if amount >= 0:
        # Blend toward a double smoothstep S-curve: stronger contrast with no hard clipping.
        target = _smoothstep(_smoothstep(x)) if amount > 0.5 else _smoothstep(x)
        t = amount * 2 if amount <= 0.5 else (amount - 0.5) * 2
        base = x if amount <= 0.5 else _smoothstep(x)
        return base + (target - base) * t
    return 0.5 + (x - 0.5) * (1 + amount)


def _saturation(rgb, amount, **_):
    y = px.luma(rgb)[..., None]
    return y + (rgb - y) * (1 + amount)


def _vibrance(rgb, amount, **_):
    y = px.luma(rgb)[..., None]
    sat = (rgb.max(-1) - rgb.min(-1))[..., None]
    return y + (rgb - y) * (1 + amount * (1 - sat))


def _temperature(rgb, amount, **_):
    lin = px.to_linear(rgb)
    gains = np.array([1 + 0.18 * amount, 1.0, 1 - 0.18 * amount], np.float32)
    return px.to_srgb(lin * gains)


def _tint(rgb, amount, **_):
    lin = px.to_linear(rgb)
    gains = np.array([1 + 0.06 * amount, 1 - 0.15 * amount, 1 + 0.06 * amount], np.float32)
    return px.to_srgb(lin * gains)


def _hue_shift(rgb, degrees, **_):
    h, s, v = px.rgb_to_hsv(np.clip(rgb, 0, 1))
    return px.hsv_to_rgb(h + degrees, s, v)


def _lum_reshape(rgb, delta_fn):
    y = px.luma(rgb)
    y2 = np.clip(y + delta_fn(np.clip(y, 0, 1)), 0, 1)
    ratio = np.where(y > 1e-4, y2 / np.maximum(y, 1e-4), 1.0)
    out = rgb * ratio[..., None]
    # Pure black pixels can't be scaled; add the delta instead.
    return np.where((y <= 1e-4)[..., None], rgb + (y2 - y)[..., None], out)


def _shadows(rgb, amount, **_):
    return _lum_reshape(rgb, lambda y: amount * y * (1 - y) ** 2 * 1.0)


def _highlights(rgb, amount, **_):
    return _lum_reshape(rgb, lambda y: amount * y * y * (1 - y) * 1.0)


def _levels(rgb, black, white, gamma, **_):
    if white - black < 0.02:
        white = black + 0.02
    x = np.clip((rgb - black) / (white - black), 0, 1)
    return x ** (1.0 / gamma)


def _auto_contrast(rgb, clip_percent, mask=None, **_):
    y = px.luma(rgb)
    sel = y[mask > 0.5] if mask is not None and (mask > 0.5).any() else y.ravel()
    lo, hi = np.percentile(sel, [clip_percent, 100 - clip_percent])
    if hi - lo < 1e-3:
        return rgb
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def _grayscale(rgb, amount, **_):
    y = px.luma(rgb)[..., None]
    return rgb + (y - rgb) * amount


def _sepia(rgb, amount, **_):
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    return rgb + (np.clip(rgb @ m.T, 0, 1) - rgb) * amount


def _invert(rgb, **_):
    return 1.0 - rgb


def _sharpen(rgb, amount, radius, threshold, **_):
    # Luminance-only unsharp mask: avoids colour fringing.
    y = px.luma(rgb)
    detail = y - px.gaussian_blur(y, radius)
    if threshold > 0:
        detail = np.where(np.abs(detail) < threshold, 0.0, detail)
    return rgb + (amount * detail)[..., None]


def _blur(rgb, radius, **_):
    return px.gaussian_blur(rgb, radius)


def _denoise(rgb, strength, **_):
    sigma = 0.8 + 2.2 * strength
    smooth = px.gaussian_blur(rgb, sigma)
    y = px.luma(smooth)
    gy, gx = np.gradient(y)
    edge = np.sqrt(gx * gx + gy * gy)
    w = strength * np.exp(-edge / 0.02)
    return rgb + (smooth - rgb) * w[..., None]


def _vignette(rgb, amount, radius, **_):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx + 0.5) / w - 0.5) ** 2 + ((yy + 0.5) / h - 0.5) ** 2) / np.sqrt(0.5)
    fall = np.clip((d - radius * 0.5) / max(1e-3, 1.0 - radius * 0.5), 0, 1)
    fall = _smoothstep(fall)
    gain = 1 + amount * fall
    return px.to_srgb(px.to_linear(rgb) * gain[..., None])


def _fill(rgb, color, opacity, **_):
    c = np.array(px.parse_color(color), np.float32)
    return rgb + (c - rgb) * opacity


def _pixelate(rgb, block, **_):
    h, w = rgb.shape[:2]
    sh, sw = max(1, h // block), max(1, w // block)
    out = np.empty_like(rgb)
    for c in range(3):
        im = Image.fromarray(rgb[..., c].astype(np.float32), "F")
        im = im.resize((sw, sh), Image.BOX).resize((w, h), Image.NEAREST)
        out[..., c] = np.asarray(im)
    return out


# ------------------------------------------------------------------- overlay

def _text(rgb, text, x, y, size, color, anchor, stroke, stroke_color, opacity, **_):
    h, w = rgb.shape[:2]
    font_px = max(6, int(round(size * h)))
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:  # pragma: no cover - very old Pillow
        font = ImageFont.load_default()
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text(
        (x * w, y * h), text, fill=255, font=font, anchor=anchor,
        stroke_width=int(round(stroke * font_px)), stroke_fill=255,
    )
    cover = np.asarray(layer, np.float32) / 255.0 * opacity
    out = rgb.copy()
    if stroke > 0:
        sl = Image.new("L", (w, h), 0)
        ImageDraw.Draw(sl).text(
            (x * w, y * h), text, fill=255, font=font, anchor=anchor,
            stroke_width=int(round(stroke * font_px)), stroke_fill=255,
        )
        inner = Image.new("L", (w, h), 0)
        ImageDraw.Draw(inner).text((x * w, y * h), text, fill=255, font=font, anchor=anchor)
        s = np.asarray(sl, np.float32) / 255.0
        i = np.asarray(inner, np.float32) / 255.0
        sc = np.array(px.parse_color(stroke_color), np.float32)
        out = out + (sc - out) * (s * opacity)[..., None]
        cover = i * opacity
    c = np.array(px.parse_color(color), np.float32)
    out = out + (c - out) * cover[..., None]
    footprint = np.asarray(layer if stroke == 0 else sl, np.float32) / 255.0
    return out, (footprint > 0).astype(np.float32)


# ----------------------------------------------------------------- geometric

def _map_layers(cv: Canvas, fn: Callable[[Image.Image, float], Image.Image], fill: str = "#000000",
                resampled: bool = False) -> Canvas:
    """Apply one geometric transform to every layer in lockstep.

    ``fn(image, fill_value)`` receives a single-channel float image. When the
    transform resamples, the touched mask is grown by the kernel footprint so
    pixels influenced by edited neighbours are not reported as untouched.
    """
    if fill == "transparent":
        fill_rgb, fill_a = (0.0, 0.0, 0.0), 0.0
    else:
        fill_rgb, fill_a = px.parse_color(fill), 1.0

    def one(a: np.ndarray, f: float) -> np.ndarray:
        return np.asarray(fn(Image.fromarray(np.ascontiguousarray(a, np.float32), "F"), f), np.float32)

    def three(a: np.ndarray) -> np.ndarray:
        return np.stack([one(a[..., c], fill_rgb[c]) for c in range(3)], -1)

    touched = one(cv.touched, 0.0)
    if resampled:
        grown = Image.fromarray(((touched > 1e-6) * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(7))
        touched = np.maximum(np.clip(touched, 0, 1), np.asarray(grown, np.float32) / 255.0)
    return Canvas(rgb=three(cv.rgb), alpha=one(cv.alpha, fill_a), ref_rgb=three(cv.ref_rgb),
                  ref_alpha=one(cv.ref_alpha, fill_a), touched=touched, notes=cv.notes)


def _crop(cv: Canvas, x0, y0, x1, y1, **_):
    w, h = cv.size
    box = (int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h)))
    box = (box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1))
    return _map_layers(cv, lambda im, f: im.crop(box))


def _resize(cv: Canvas, width, height, scale, **_):
    w, h = cv.size
    if scale and scale > 0:
        nw, nh = round(w * scale), round(h * scale)
    elif width and height:
        nw, nh = width, height
    elif width:
        nw, nh = width, round(h * width / w)
    else:
        nw, nh = round(w * height / h), height
    nw, nh = max(1, int(nw)), max(1, int(nh))
    return _map_layers(cv, lambda im, f: im.resize((nw, nh), Image.LANCZOS), resampled=True)


def _inscribed(w: float, h: float, deg: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle inside a w*h rectangle rotated by deg."""
    a = np.radians(deg)
    sin_a, cos_a = abs(np.sin(a)), abs(np.cos(a))
    long_side, short_side = (w, h) if w >= h else (h, w)
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def _rotate(cv: Canvas, degrees, mode, fill, **_):
    d = degrees % 360
    exact = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
    if round(d, 6) in exact:
        t = exact[round(d, 6)]
        return _map_layers(cv, lambda im, f: im.transpose(t))
    if round(d, 6) == 0:
        return cv
    w, h = cv.size
    out = _map_layers(cv, lambda im, f: im.rotate(degrees, Image.BICUBIC, expand=(mode == "expand"), fillcolor=f),
                      fill=fill, resampled=True)
    if mode == "crop":
        wr, hr = _inscribed(w, h, degrees)
        W, H = out.size
        box = (int(np.ceil((W - wr) / 2)), int(np.ceil((H - hr) / 2)),
               int(np.floor((W + wr) / 2)), int(np.floor((H + hr) / 2)))
        out = _map_layers(out, lambda im, f: im.crop(box))
    return out


def _flip(cv: Canvas, direction, **_):
    t = Image.FLIP_LEFT_RIGHT if direction == "horizontal" else Image.FLIP_TOP_BOTTOM
    return _map_layers(cv, lambda im, f: im.transpose(t))


def _pad(cv: Canvas, top, right, bottom, left, color, **_):
    w, h = cv.size

    def expand(im: Image.Image, f):
        canvas = Image.new("F", (w + left + right, h + top + bottom), f)
        canvas.paste(im, (left, top))
        return canvas

    return _map_layers(cv, expand, fill=color)


# ------------------------------------------------------------------- catalog

def F(name, default, lo, hi, desc="", required=False):
    return Param(name, "float", default, lo, hi, desc=desc, required=required)


def I(name, default, lo, hi, desc="", required=False):
    return Param(name, "int", default, lo, hi, desc=desc, required=required)


CLIP = "clipping of highlights/shadows at strong values"
BAND = "banding/posterization in smooth gradients (sky, skin) at strong values on 8-bit sources"
NOISE = "amplifies existing sensor noise and JPEG blocking in dark areas"
SEAM = "visible seam at region edge if feather is 0"

OPS: dict[str, Op] = {o.name: o for o in [
    Op("exposure", "tonal", "Photographic exposure in linear light", (
        F("stops", 0.0, -3, 3, "EV; +1 doubles light", True),), _exposure, (CLIP, NOISE)),
    Op("brightness", "tonal", "Midtone brightness curve; black and white points fixed", (
        F("amount", 0.0, -1, 1, "+ brighter", True),), _brightness, (BAND,)),
    Op("contrast", "tonal", "S-curve contrast (no hard clip when +)", (
        F("amount", 0.0, -1, 1, "+ more contrast", True),), _contrast, (BAND, "crushes shadow/highlight detail at strong values")),
    Op("saturation", "tonal", "Global colour saturation", (
        F("amount", 0.0, -1, 1, "-1 = grayscale", True),), _saturation, ("channel clipping/gamut posterization in already saturated colours",)),
    Op("vibrance", "tonal", "Saturation that spares already-saturated colours", (
        F("amount", 0.0, -1, 1, required=True),), _vibrance, ()),
    Op("temperature", "tonal", "White balance blue<->amber", (
        F("amount", 0.0, -1, 1, "+ warmer", True),), _temperature, ("colour cast in neutrals",)),
    Op("tint", "tonal", "White balance green<->magenta", (
        F("amount", 0.0, -1, 1, "+ magenta", True),), _tint, ("colour cast in neutrals",)),
    Op("hue_shift", "tonal", "Rotate hues", (
        F("degrees", 0.0, -180, 180, required=True),), _hue_shift, ("unnatural skin/sky colours; restrict with a color_range",)),
    Op("shadows", "tonal", "Lift (+) or deepen (-) shadows only", (
        F("amount", 0.0, -1, 1, required=True),), _shadows, (NOISE, "halo-free but flattens local contrast")),
    Op("highlights", "tonal", "Recover (-) or boost (+) highlights only", (
        F("amount", 0.0, -1, 1, required=True),), _highlights, ("cannot recover detail already clipped to pure white",)),
    Op("levels", "tonal", "Input black/white points and gamma", (
        F("black", 0.0, 0, 0.5), F("white", 1.0, 0.5, 1), F("gamma", 1.0, 0.2, 5, ">1 brighter mids")),
        _levels, (CLIP, BAND)),
    Op("auto_contrast", "tonal", "Stretch luminance range to full scale", (
        F("clip_percent", 0.5, 0, 5),), _auto_contrast, (CLIP, "colour shift if one channel dominates")),
    Op("grayscale", "tonal", "Desaturate to luminance", (
        F("amount", 1.0, 0, 1),), _grayscale, ()),
    Op("sepia", "tonal", "Sepia tone", (F("amount", 1.0, 0, 1),), _sepia, ()),
    Op("invert", "tonal", "Negative", (), _invert, ()),
    Op("sharpen", "tonal", "Luminance unsharp mask", (
        F("amount", 0.6, 0, 3), F("radius", 1.2, 0.3, 10, "px"), F("threshold", 0.0, 0, 0.2)),
        _sharpen, ("halos along high-contrast edges at amount>1 or radius>3", "amplifies noise/JPEG artifacts unless threshold>0",
                   "cannot restore detail lost to motion/defocus blur")),
    Op("blur", "tonal", "Gaussian blur", (F("radius", 2.0, 0.3, 100, "px sigma", True),), _blur, (SEAM,)),
    Op("denoise", "tonal", "Edge-aware smoothing", (F("strength", 0.5, 0, 1, required=True),), _denoise,
       ("waxy/plastic texture and loss of fine detail at strength>0.6",)),
    Op("vignette", "tonal", "Darken (-) or lighten (+) edges", (
        F("amount", -0.3, -1, 1, required=True), F("radius", 0.8, 0.2, 1.5, "clear centre size")), _vignette, (BAND,)),
    Op("fill", "tonal", "Solid colour over the region (redaction, recolour)", (
        Param("color", "color", "#000000", required=True), F("opacity", 1.0, 0, 1)), _fill, (SEAM,)),
    Op("pixelate", "tonal", "Mosaic the region (redaction)", (I("block", 16, 2, 256, "px"),), _pixelate, ()),
    Op("text", "overlay", "Draw text; x,y is the anchor point (normalized)", (
        Param("text", "str", "", required=True), F("x", 0.5, 0, 1), F("y", 0.5, 0, 1),
        F("size", 0.06, 0.01, 0.5, "font height / image height"), Param("color", "color", "#ffffff"),
        Param("anchor", "enum", "mm", choices=("lt", "mt", "rt", "lm", "mm", "rm", "lb", "mb", "rb", "ls", "ms", "rs"),
              desc="PIL anchor: l/m/r + t/m/b/s"),
        F("stroke", 0.0, 0, 0.2, "outline width / font height"), Param("stroke_color", "color", "#000000"),
        F("opacity", 1.0, 0, 1)), _text,
       ("uses a bundled sans-serif font; cannot match existing typography", "text may run off-canvas if long")),
    Op("crop", "geometric", "Crop to normalized box", (
        F("x0", 0.0, 0, 1), F("y0", 0.0, 0, 1), F("x1", 1.0, 0, 1), F("y1", 1.0, 0, 1)), _crop, (), lossless=True),
    Op("resize", "geometric", "Lanczos resize: give scale, or width and/or height (px; aspect kept if one)", (
        I("width", 0, 0, 30000), I("height", 0, 0, 30000), F("scale", 0.0, 0, 8)), _resize,
       ("upscaling cannot invent detail: result is soft (>1.5x) or visibly blurry (>2x)", "slight ringing near hard edges",
        "downscaling discards detail permanently")),
    Op("rotate", "geometric", "Rotate counter-clockwise; 90/180/270 are lossless", (
        F("degrees", 0.0, -360, 360, required=True),
        Param("mode", "enum", "expand", choices=("expand", "crop", "same"),
              desc="expand canvas | crop to largest clean rectangle | keep size"),
        Param("fill", "color", "#000000", desc="corner fill colour or 'transparent'")), _rotate,
       ("non-right angles resample every pixel (slight softening)", "'expand'/'same' leave filled corners; 'crop' loses edge content")),
    Op("flip", "geometric", "Mirror", (Param("direction", "enum", "horizontal", choices=("horizontal", "vertical"), required=True),),
       _flip, ("mirrors any text or logos in the image",), lossless=True),
    Op("pad", "geometric", "Add border (px)", (
        I("top", 0, 0, 10000), I("right", 0, 0, 10000), I("bottom", 0, 0, 10000), I("left", 0, 0, 10000),
        Param("color", "color", "#ffffff", desc="or 'transparent'")), _pad, (), lossless=True),
]}


def catalog_text() -> str:
    """Compact, deterministic catalog for the planner prompt (stable => cacheable)."""
    lines = []
    for kind in ("tonal", "overlay", "geometric"):
        lines.append(f"[{kind}]")
        for op in OPS.values():
            if op.kind != kind:
                continue
            ps = []
            for p in op.params:
                if p.type in ("float", "int"):
                    spec = f"{p.min:g}..{p.max:g}"
                elif p.type == "enum":
                    spec = "|".join(p.choices)
                else:
                    spec = p.type
                req = "*" if p.required else f"={p.default}"
                ps.append(f"{p.name}{req}:{spec}" + (f"({p.desc})" if p.desc else ""))
            lines.append(f"{op.name}({', '.join(ps)}) - {op.summary}")
    return "\n".join(lines)
