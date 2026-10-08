"""Execution, scope lock, verification and artifact measurement.

The edit runs in float32. At the end, every pixel whose combined edit mask is
exactly zero is copied from the (geometrically transformed) source, so it is
byte-identical to the input; this is checked, not assumed.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .ops import OPS, Canvas
from .plan import Plan, Step
from .regions import build_mask

# ------------------------------------------------------------------- loading


@dataclass
class Source:
    path: str
    format: str
    mode: str
    rgb: np.ndarray  # float32 HxWx3, displayed orientation
    alpha: Optional[np.ndarray]  # float32 HxW or None
    u8: np.ndarray  # uint8 HxWx3 reference for verification
    exif: Optional[Image.Exif]
    icc: Optional[bytes]
    dpi: Optional[tuple]
    jpeg_quality: Optional[int]
    jpeg_subsampling: Optional[int]
    grayscale: bool
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


_STD_LUMA_Q = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]


def estimate_jpeg_quality(im: Image.Image) -> Optional[int]:
    """Estimate IJG quality from the luminance quantization table."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    table = sorted(q[0])
    best, best_err = None, float("inf")
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - quality * 2
        std = sorted(min(255, max(1, (v * scale + 50) // 100)) for v in _STD_LUMA_Q)
        err = sum(abs(a - b) for a, b in zip(table, std))
        if err < best_err:
            best, best_err = quality, err
    return best


def load(path: str) -> Source:
    im = Image.open(path)
    im.load()
    fmt = (im.format or os.path.splitext(path)[1].lstrip(".")).upper()
    notes: list[str] = []
    jpeg_q = estimate_jpeg_quality(im) if fmt == "JPEG" else None
    sub = None
    if fmt == "JPEG":
        try:
            from PIL import JpegImagePlugin
            sub = JpegImagePlugin.get_sampling(im)
        except Exception:  # pragma: no cover
            sub = None
    exif = im.getexif()
    orientation = exif.get(0x0112, 1) if exif else 1
    if orientation not in (1, None):
        im = ImageOps.exif_transpose(im)
        notes.append(f"EXIF orientation {orientation} applied to pixels; tag reset to 1 (display unchanged)")
    icc = im.info.get("icc_profile")
    dpi = im.info.get("dpi")
    mode = im.mode
    grayscale = mode in ("1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F")

    if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        a = np.asarray(im, np.float32)
        a = a / (65535.0 if a.max() > 255 else 255.0)
        rgb = np.repeat(np.clip(a, 0, 1)[..., None], 3, -1)
        alpha = None
        notes.append(f"source is high bit depth ({mode}); output will be 8-bit")
    else:
        if mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            notes.append("palette image converted to full colour; output is saved as full colour")
        elif mode == "CMYK":
            im = im.convert("RGB")
            notes.append("CMYK source converted to RGB; colours may shift slightly")
        elif mode == "1":
            im = im.convert("L")
        elif mode not in ("RGB", "RGBA", "L", "LA"):
            im = im.convert("RGBA" if "A" in mode else "RGB")
        has_alpha = im.mode in ("RGBA", "LA")
        alpha = np.asarray(im.getchannel("A"), np.float32) / 255.0 if has_alpha else None
        base = im.convert("RGB")
        rgb = np.asarray(base, np.float32) / 255.0
    u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return Source(path, fmt, mode, rgb, alpha, u8, exif if len(exif) else None, icc, dpi, jpeg_q, sub,
                  grayscale, notes)


def preview_jpeg(src: Source, max_side: int) -> tuple[bytes, tuple[int, int]]:
    """Downscaled JPEG for the planner - the main input-token lever."""
    im = Image.fromarray(src.u8, "RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return buf.getvalue(), im.size


def image_facts(src: Source) -> str:
    y = src.u8.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    clip_hi = float((src.u8 == 255).any(-1).mean())
    clip_lo = float((src.u8 == 0).any(-1).mean())
    p = np.percentile(y, [1, 50, 99])
    sat = float((src.u8.max(-1).astype(int) - src.u8.min(-1)).mean() / 255)
    w, h = src.size
    return (f"{w}x{h} {src.format} {src.mode}{' +alpha' if src.alpha is not None else ''}; "
            f"luma p1/p50/p99={p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}; clipped white {clip_hi:.1%}, black {clip_lo:.1%}; "
            f"mean saturation {sat:.2f}")


# ----------------------------------------------------------------- execution


@dataclass
class StepStats:
    coverage: float  # mean mask weight over the canvas at that step
    color_selected: Optional[float] = None


@dataclass
class Result:
    rgb: np.ndarray  # uint8 HxWx3
    alpha: Optional[np.ndarray]  # uint8 HxW or None
    ref: np.ndarray  # uint8 source after geometric ops only
    touched: np.ndarray  # float HxW
    step_stats: list[StepStats]
    max_drift_outside: int  # largest float->8bit deviation outside the scope before the lock
    untouched_identical: bool
    geometric: bool


def _quantize(a: np.ndarray, dither: Optional[np.ndarray] = None) -> np.ndarray:
    x = a * 255.0
    if dither is not None:
        x = x + dither
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def execute(src: Source, plan: Plan, dither: bool = False, seed: int = 0) -> Result:
    alpha = src.alpha if src.alpha is not None else np.ones(src.rgb.shape[:2], np.float32)
    cv = Canvas(rgb=src.rgb.copy(), alpha=alpha.copy(), ref_rgb=src.rgb, ref_alpha=alpha,
                touched=np.zeros(src.rgb.shape[:2], np.float32))
    stats: list[StepStats] = []
    geometric = False
    for step in plan.steps:
        op = OPS[step.op]
        if op.kind == "geometric":
            geometric = True
            cv = op.fn(cv, **step.params)
            stats.append(StepStats(coverage=1.0))
            continue
        mask = build_mask(step.region, cv.rgb)
        out = op.fn(cv.rgb, mask=mask, **step.params)
        footprint = None
        if isinstance(out, tuple):
            out, footprint = out
        m = np.ones(cv.rgb.shape[:2], np.float32) if mask is None else mask
        if footprint is not None:
            m = m * footprint
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        cv.rgb = np.clip(cv.rgb + (out - cv.rgb) * m[..., None], 0.0, 1.0).astype(np.float32)
        cv.touched = np.maximum(cv.touched, m)
        sel = None
        if step.region.color_range is not None:
            sel = float((m > 0.5).mean())
        stats.append(StepStats(coverage=float(m.mean()), color_selected=sel))

    ref = _quantize(cv.ref_rgb)
    noise = None
    if dither:
        rng = np.random.default_rng(seed)
        noise = (rng.random(cv.rgb.shape, np.float32) - rng.random(cv.rgb.shape, np.float32)) * (cv.touched[..., None] > 0)
    rgb = _quantize(cv.rgb, noise)
    outside = cv.touched <= 0.0
    drift = int(np.abs(rgb[outside].astype(int) - ref[outside]).max()) if outside.any() else 0
    rgb[outside] = ref[outside]  # scope lock

    out_alpha = None
    if src.alpha is not None or (cv.alpha < 1.0).any():
        out_alpha = _quantize(cv.alpha)

    identical = True
    if not geometric:
        identical = bool(np.array_equal(rgb[outside], src.u8[outside]))
        if out_alpha is not None and src.alpha is not None:
            identical &= bool(np.array_equal(out_alpha, _quantize(src.alpha)))
    else:
        identical = bool(np.array_equal(rgb[outside], ref[outside]))
    return Result(rgb, out_alpha, ref, cv.touched, stats, drift, identical, geometric)


# ------------------------------------------------------------------ analysis

_MILD = {
    "exposure": ("stops", 0.75), "brightness": ("amount", 0.4), "contrast": ("amount", 0.4),
    "saturation": ("amount", 0.4), "vibrance": ("amount", 0.6), "temperature": ("amount", 0.4),
    "tint": ("amount", 0.4), "hue_shift": ("degrees", 25), "shadows": ("amount", 0.6),
    "highlights": ("amount", 0.6), "denoise": ("strength", 0.6), "vignette": ("amount", 0.5),
}


def _is_strong(step: Step) -> bool:
    p = step.params
    if step.op in _MILD:
        k, lim = _MILD[step.op]
        return abs(p[k]) > lim
    if step.op == "sharpen":
        return p["amount"] > 1.0 or p["radius"] > 3.0
    if step.op == "levels":
        return p["black"] > 0.08 or p["white"] < 0.92 or not 0.7 <= p["gamma"] <= 1.4
    if step.op == "resize":
        return True
    if step.op == "rotate":
        return round(p["degrees"] % 90, 6) != 0
    return False


def analyse(src: Source, plan: Plan, res: Result, out_format: str) -> tuple[list[str], list[str]]:
    """Return (measured findings, possible-risk notes). Measured = observed in the actual result."""
    measured: list[str] = []
    possible: list[str] = []
    h, w = res.rgb.shape[:2]
    touched = res.touched > 0
    frac = float(touched.mean())
    changed = float((res.rgb != res.ref).any(-1).mean())

    if not res.untouched_identical:  # pragma: no cover - would be an engine bug
        measured.append("ERROR: pixels outside the edit scope differ from the source - do not use this output")

    for i, (step, st) in enumerate(zip(plan.steps, res.step_stats), 1):
        op = OPS[step.op]
        if op.kind != "geometric" and st.coverage < 0.0005:
            measured.append(f"step {i} ({step.op}): region selects almost nothing ({st.coverage:.3%} of the image) - "
                            "the selection probably missed the intended subject")
        if st.color_selected is not None:
            measured.append(f"step {i} ({step.op}): colour selection matched {st.color_selected:.1%} of the image; "
                            "similar colours elsewhere inside the region are affected too")
        if op.kind == "tonal" and not step.region.is_full and step.region.feather == 0 \
                and step.region.color_range is None and step.op not in ("fill", "pixelate"):
            possible.append(f"step {i} ({step.op}): hard region edge (feather 0) may show a visible seam")
        if step.region.shape != "full":
            possible.append(f"step {i} ({step.op}): region boundary was estimated from a downscaled preview; "
                            "it can be off by a few percent of the image size")
        for r in op.risks:
            level = "elevated" if _is_strong(step) else "low"
            if step.op == "resize":
                scale = _resize_factor(step, src, res)
                if scale <= 1.0 and r.startswith("upscaling"):
                    continue
                if scale > 1.0 and r.startswith("downscaling"):
                    continue
                if scale > 1.5 and r.startswith("upscaling"):
                    measured.append(f"step {i}: upscaled {scale:.2f}x - {r}")
                    continue
            if step.op == "rotate" and round(step.params["degrees"] % 90, 6) == 0:
                continue
            possible.append(f"step {i} ({step.op}) [{level}]: {r}")

    if touched.any() and not res.geometric:
        t = touched
        new_clip = ((res.rgb[t] == 255).any(-1) & ~(res.ref[t] == 255).any(-1)) | \
                   ((res.rgb[t] == 0).any(-1) & ~(res.ref[t] == 0).any(-1))
        clip = float(new_clip.mean())
        if clip > 0.005:
            measured.append(f"newly clipped pixels in the edited area: {clip:.1%} (lost highlight/shadow detail)")
        if t.mean() > 0.01:
            lw = np.array([0.2126, 0.7152, 0.0722], np.float32)
            yb = np.round(res.ref[t].astype(np.float32) @ lw).astype(int)
            ya = np.round(res.rgb[t].astype(np.float32) @ lw).astype(int)

            def gaps(y):
                lo, hi = np.percentile(y, [1, 99]).astype(int)
                if hi - lo < 16:
                    return 0.0
                hist = np.bincount(y, minlength=256)[lo:hi + 1]
                return float((hist == 0).mean())

            gb, ga = gaps(yb), gaps(ya)
            if ga > 0.15 and ga > gb + 0.1:
                measured.append(f"histogram combing: {ga:.0%} of tonal levels empty after edit (was {gb:.0%}) - "
                                "banding likely in smooth gradients; re-run with --dither to mask it")

    if src.format == "JPEG" and out_format == "JPEG":
        possible.append("JPEG re-encode adds a small generational loss to edited and unedited areas alike "
                        "(save as PNG for a lossless result)")
    if res.alpha is not None and out_format == "JPEG" and (res.alpha < 255).any():
        measured.append("output format JPEG has no transparency: transparent areas will be flattened onto white")
    if src.icc and b"sRGB" not in src.icc[:200]:
        possible.append("source has a non-sRGB colour profile; adjustments are applied to encoded values and the "
                        "profile is kept, so strong colour edits may look slightly different than on sRGB images")
    possible.extend(src.notes)

    summary = (f"scope: {frac:.1%} of the output canvas may change; {changed:.1%} of pixels actually changed; "
               f"untouched pixels identical to source: {'yes' if res.untouched_identical else 'NO'}")
    measured.insert(0, summary)
    return measured, possible


def _resize_factor(step: Step, src: Source, res: Result) -> float:
    p = step.params
    if p["scale"]:
        return float(p["scale"])
    w, h = src.size
    if p["width"]:
        return p["width"] / w if not p["height"] else max(p["width"] / w, p["height"] / h)
    return p["height"] / h


# --------------------------------------------------------------------- saving

_EXT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF",
        ".bmp": "BMP", ".gif": "GIF"}


def format_for(path: str, fallback: str) -> str:
    return _EXT.get(os.path.splitext(path)[1].lower(), fallback)


def to_image(src: Source, res: Result, out_format: str) -> Image.Image:
    gray = src.grayscale and bool((res.rgb[..., 0] == res.rgb[..., 1]).all() and (res.rgb[..., 1] == res.rgb[..., 2]).all())
    if gray:
        im = Image.fromarray(res.rgb[..., 0], "L")
    else:
        im = Image.fromarray(res.rgb, "RGB")
    if res.alpha is not None:
        a = Image.fromarray(res.alpha, "L")
        if out_format == "JPEG":
            bg = Image.new(im.mode, im.size, 255 if gray else (255, 255, 255))
            bg.paste(im, mask=a)
            im = bg
        else:
            im = im.convert("LA" if gray else "RGBA")
            im.putalpha(a)
    return im


def save(src: Source, res: Result, path: str, quality: Optional[int] = None) -> dict:
    out_format = format_for(path, src.format)
    im = to_image(src, res, out_format)
    kw: dict = {}
    if src.exif is not None and out_format in ("JPEG", "PNG", "WEBP", "TIFF"):
        exif = src.exif
        if 0x0112 in exif:
            exif[0x0112] = 1
        kw["exif"] = exif.tobytes()
    if src.icc and out_format in ("JPEG", "PNG", "WEBP", "TIFF"):
        kw["icc_profile"] = src.icc
    if src.dpi and out_format in ("JPEG", "PNG", "TIFF"):
        kw["dpi"] = src.dpi
    if out_format == "JPEG":
        kw["quality"] = quality or max(92, src.jpeg_quality or 0)
        if src.jpeg_subsampling is not None and src.jpeg_subsampling >= 0:
            kw["subsampling"] = src.jpeg_subsampling
        kw["optimize"] = True
    elif out_format == "WEBP":
        kw["lossless"] = quality is None
        if quality:
            kw["quality"] = quality
    elif out_format == "PNG":
        kw["optimize"] = True
    im.save(path, out_format, **kw)
    return {"format": out_format, **{k: v for k, v in kw.items() if k in ("quality", "subsampling", "lossless")}}
