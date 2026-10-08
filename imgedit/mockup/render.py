"""Photoreal compositing of a logo onto a product photo.

Physical model (linear light): photo = albedo_product * illumination. A
print replaces the albedo, so  out = albedo_ink * photo / albedo_product,
where albedo_product is estimated robustly around the print. This carries
the photo's real shading, folds, colour of light and vignetting onto the
ink. On top of that: fabric weave texture, glossy highlights (which sit
*over* ink), ink opacity, lens softness and sensor grain matched to the
photo. Pixels outside the logo footprint are byte-identical to the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..pixels import parse_color, to_linear, to_srgb
from .geometry import Cylinder, Flat, Mapping, Placement, fold_displacement, map_cylinder, map_flat

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass
class Look:
    technique: str = "print"  # print | engrave | emboss | deboss
    opacity: float = 0.97  # ink hiding power (fabric ~0.92-0.96)
    gloss: float = 0.0  # 0 matte .. 1 glossy: how much photo highlights sit over the ink
    texture: float = 0.0  # 0..1 how much substrate micro-texture shows through
    engrave_color: str = "#3a2a1e"  # tone of engraved material (wood burn default)
    relief: float = 0.6  # emboss/deboss strength
    blur: Optional[float] = None  # None = match photo softness automatically
    grain: Optional[float] = None  # None = match photo noise automatically


@dataclass
class Composite:
    rgb: np.ndarray  # uint8 full image
    footprint: np.ndarray  # float HxW final alpha
    mapping: Mapping
    stats: dict = field(default_factory=dict)


def _lum(a: np.ndarray) -> np.ndarray:
    return a @ LUM


def sample_logo(logo: np.ndarray, m: Mapping) -> np.ndarray:
    """Premultiplied LINEAR-light RGBA logo sampled through the inverse map, box-filtered to output size.

    Filtering in linear light matters for thin bright strokes on dark products: a white line that
    covers 25% of a pixel must average to 25% of the light (sRGB ~137), not to sRGB 64.
    """
    prem = logo.copy()
    prem[..., :3] = to_linear(np.clip(prem[..., :3], 0, 1)) * prem[..., 3:4]
    # Pre-filter (mip-map): a 2400px logo sampled straight into a 35px footprint aliases - thin strokes
    # fall between samples and break up. Area-average it down to ~2x the supersampled footprint first.
    target_w = max(8, int(m.footprint_px[0] * m.ss * 2))
    if prem.shape[1] > target_w * 1.5:
        f = target_w / prem.shape[1]
        prem = cv2.resize(prem, (target_w, max(2, int(round(prem.shape[0] * f)))), interpolation=cv2.INTER_AREA)
    lh, lw = prem.shape[:2]
    mx = m.lu * (lw - 1)
    my = m.lv * (lh - 1)
    s = cv2.remap(prem, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s[~m.valid] = 0
    x0, y0, x1, y1 = m.roi
    return cv2.resize(s, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)


def _sharpness(a: np.ndarray, mask: np.ndarray) -> float:
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    b = cv2.GaussianBlur(a, (0, 0), 1.0)
    gb = np.sqrt(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    sel = mask & (g > np.percentile(g[mask], 90)) if mask.any() else mask
    if sel.sum() < 30:
        return 1.0
    return float(g[sel].mean() / max(gb[sel].mean(), 1e-6))


def estimate_softness(photo_lum: np.ndarray, roi, alpha: np.ndarray) -> float:
    """Gaussian sigma (px) that makes crisp logo edges as soft as real edges near the print.

    Only edges INSIDE the product count: in cut-out product shots the silhouette against the white
    background is feathered by the background removal, not by the lens.
    """
    x0, y0, x1, y1 = roi
    h, w = photo_lum.shape
    pad = max(20, int(0.5 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = photo_lum[Y0:Y1, X0:X1]
    backdrop = crop > 0.9
    interior = ~(cv2.dilate(backdrop.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if interior.sum() < 200:
        return 0.5
    target = _sharpness(crop, interior)
    edge = alpha > 0.02
    if edge.sum() < 30:
        return 0.5
    best, best_err = 0.0, 1e9
    for sig in (0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.6):
        a = cv2.GaussianBlur(alpha, (0, 0), sig) if sig > 0 else alpha
        sh = _sharpness(a, cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        err = abs(sh - target)
        if err < best_err:
            best, best_err = sig, err
    return best


def estimate_noise(photo: np.ndarray, roi) -> float:
    """Sensor noise sigma (0..1 units) from the flattest patches around the print."""
    x0, y0, x1, y1 = roi
    crop = _lum(photo[y0:y1, x0:x1])
    if crop.size < 400:
        return 0.0
    hp = crop - cv2.GaussianBlur(crop, (0, 0), 1.0)
    local = cv2.GaussianBlur(np.abs(hp), (0, 0), 4)
    flat = local < np.percentile(local, 25)
    return float(1.4826 * np.median(np.abs(hp[flat]))) * 1.6 if flat.any() else 0.0


def composite(photo: np.ndarray, logo: np.ndarray, surface, pl: Placement, look: Look,
              occluder: Optional[np.ndarray] = None, ss: int = 3, seed: int = 0) -> Composite:
    """photo: HxWx3 float sRGB 0..1; logo: hxwx4 float sRGB 0..1 (straight alpha)."""
    h, w = photo.shape[:2]
    aspect = logo.shape[1] / logo.shape[0]
    plum = _lum(photo).astype(np.float32)
    if surface.kind == "cylinder":
        m = map_cylinder(surface, pl, aspect, (h, w), ss)
    else:
        disp = None
        if surface.fabric:
            q = np.asarray(surface.quad, float)
            scale = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            disp = fold_displacement(plum, scale)
        m = map_flat(surface, pl, aspect, (h, w), ss, disp)
    x0, y0, x1, y1 = m.roi
    if x1 <= x0 or y1 <= y0:
        raise ValueError("logo placement falls outside the photo")
    s = sample_logo(logo, m)
    alpha = np.clip(s[..., 3], 0, 1)
    ink_l = np.where(alpha[..., None] > 1e-4, s[..., :3] / np.maximum(alpha[..., None], 1e-4), 0)  # linear
    if occluder is not None:
        alpha = alpha * (1 - occluder[y0:y1, x0:x1])

    blur = look.blur if look.blur is not None else estimate_softness(plum, m.roi, alpha)
    if blur > 0:
        a2 = cv2.GaussianBlur(alpha, (0, 0), blur)
        prem = cv2.GaussianBlur(ink_l * alpha[..., None], (0, 0), blur)
        ink_l = np.where(a2[..., None] > 1e-4, prem / np.maximum(a2[..., None], 1e-4), 0)
        alpha = a2

    P = photo[y0:y1, x0:x1].astype(np.float32)
    Pl = to_linear(P)
    # Product albedo: robust colour of the bare surface under and around the print, taken from
    # a heavily smoothed image so folds/edges don't bias it.
    cover = alpha > 0.05
    ring = cv2.dilate(cover.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    sel = ring if ring.sum() > 50 else np.ones_like(ring)
    # "Full light" reference = the bright end of the bare surface (90th pct luminance), not the median:
    # a median is pulled down by the shaded side of curved products and would over-brighten the ink.
    # On dark products the photo's luminance is dominated by sensor/JPEG noise and fabric sheen, not by
    # illumination, so shading is taken at a coarser scale and its strength is reduced (confidence k).
    smooth = cv2.GaussianBlur(Pl, (0, 0), 1.2)
    lum_s = _lum(smooth)
    ref_l = max(float(np.percentile(lum_s[sel], 90)), 1e-4)
    base = np.median(Pl[sel & (lum_s >= np.percentile(lum_s[sel], 75))], 0) + 1e-4
    k_conf = float(np.clip(np.sqrt(ref_l / 0.04), 0.35, 1.0))
    sig_sh = 1.2 if k_conf >= 1.0 else max(1.2, m.footprint_px[0] / 50)
    lum_sh = _lum(cv2.GaussianBlur(Pl, (0, 0), sig_sh))
    raw = np.clip(lum_sh / ref_l, 0.03, 1.15)
    shade_l = np.clip(1 + (raw - 1) * k_conf, 0.03, 1.15)[..., None]
    light_tint = base / max(float(_lum(base[None])[0]), 1e-4)
    is_neutral = np.ptp(light_tint) < 0.25  # neutral product: its cast is the light's colour
    # A bright neutral product's cast is the light's colour; a dark product's cast is its dye, so no tint.
    tint = np.clip(light_tint, 0.7, 1.3) if (is_neutral and ref_l > 0.15) else np.ones(3, np.float32)
    stats = {"blur_sigma": blur, "base_srgb": to_srgb(base).round(3).tolist(), "shading_confidence": round(k_conf, 2)}

    tech = look.technique
    if tech == "engrave":
        ink_l = np.broadcast_to(to_linear(np.array(parse_color(look.engrave_color), np.float32)), ink_l.shape).copy()
    if tech in ("print", "engrave"):
        out_l = np.clip(ink_l, 0, 1) * shade_l * tint
        if look.texture > 0:
            hp = _lum(P) - cv2.GaussianBlur(_lum(P), (0, 0), 1.0)
            out_l = to_linear(np.clip(to_srgb(out_l) + (hp * look.texture)[..., None], 0, 1))
        a = alpha * look.opacity
        res = to_srgb(Pl + (out_l - Pl) * a[..., None])  # coverage blends in linear light
    else:  # emboss / deboss: no ink, only relief shading of the substrate
        d = 1.0 if tech == "emboss" else -1.0
        k = max(1.0, (m.footprint_px[0] / 400.0))
        gy, gx = np.gradient(cv2.GaussianBlur(alpha, (0, 0), k))
        light = (-gx - gy) * d * look.relief * 6.0
        res_l = Pl * (1 + light[..., None])
        res = to_srgb(res_l)
        res = P + (res - P) * np.clip(np.abs(light) * 50, 0, 1)[..., None]
        alpha = np.clip(np.abs(light) * 50, 0, 1) * (cv2.dilate(cover.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

    if look.gloss > 0 and tech != "deboss":
        # Specular reflection is light ADDED on top of whatever albedo is there (ink or glaze):
        # excess = photo - diffuse shading, both in linear light. Continuous, so no threshold speckle.
        lin_l = cv2.GaussianBlur(_lum(Pl), (0, 0), 1.5)
        diffuse = cv2.GaussianBlur(lin_l, (0, 0), max(4.0, m.footprint_px[0] / 12))
        excess = np.clip(lin_l - diffuse * 1.04, 0, None) * look.gloss
        res = to_srgb(to_linear(res) + (excess * (alpha > 0))[..., None] * tint)
        stats["gloss_px"] = int((excess > 0.02).sum())

    grain = look.grain if look.grain is not None else estimate_noise(photo, m.roi)
    if grain > 0:
        rng = np.random.default_rng(seed)
        res = res + (rng.standard_normal(res.shape[:2]).astype(np.float32) * grain)[..., None] * (alpha > 0.01)[..., None]
    stats["grain_sigma"] = round(float(grain), 4)

    res = np.clip(res, 0, 1)
    out = (np.clip(np.round(photo * 255), 0, 255)).astype(np.uint8)
    region = np.clip(np.round(res * 255), 0, 255).astype(np.uint8)
    touched = alpha > 1e-3
    sub = out[y0:y1, x0:x1]
    sub[touched] = region[touched]
    foot = np.zeros((h, w), np.float32)
    foot[y0:y1, x0:x1] = np.where(touched, alpha, 0)
    stats["shade_range"] = [round(float(np.percentile(shade_l[cover], 2)), 2),
                            round(float(np.percentile(shade_l[cover], 98)), 2)] if cover.any() else [1, 1]
    stats["ink_mean_srgb"] = to_srgb(ink_l[cover]).mean(0).round(3).tolist() if cover.any() else None
    return Composite(out, foot, m, stats)
