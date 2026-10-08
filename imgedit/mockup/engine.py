"""Mockup execution: refine geometry, enforce physical size, composite, verify, measure."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import engine as base_engine
from .geometry import Cylinder, guide_lines, map_cylinder, map_flat
from .logo import Logo, has_opaque_light_background, knockout_white, recolor
from .plan import MockupPlan, PlacementSpec
from .refine import refine_cylinder, snap_quad
from .render import composite

LIMITS = (
    "Composite, not a photo of a real print: ink texture, thickness and colour come from a physical model of the photo's own light and surface.",
    "Screen colours are not print colours: neon, metallic, glitter, foil and exact Pantone matches can't be shown faithfully.",
    "Embroidery, patches and 3D effects are not simulated (stitches can't be invented without generating detail).",
    "Nothing in the photo is removed: an existing print or label on the product stays under/around the logo.",
    "Pixels outside the logo footprint are locked byte-for-byte to the product photo (verified).",
)


@dataclass
class MockupResult:
    rgb: np.ndarray
    footprint: np.ndarray
    per_placement: list[dict] = field(default_factory=list)
    identical_outside: bool = True


def _lab(rgb: np.ndarray) -> np.ndarray:
    a = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]


def prepare_logo(logo: Logo, spec: PlacementSpec) -> Logo:
    lg = copy.deepcopy(logo)
    if spec.ink == "knockout_white":
        lg = knockout_white(lg)
    elif spec.ink == "single_color" and spec.ink_color:
        lg = recolor(lg, spec.ink_color)
    return lg


def _mapping(spec: PlacementSpec, aspect: float, shape):
    if isinstance(spec.surface, Cylinder):
        return map_cylinder(spec.surface, spec.placement, aspect, shape, 1)
    return map_flat(spec.surface, spec.placement, aspect, shape, 1)


def enforce_size(spec: PlacementSpec, aspect: float, shape, notes: list[str]) -> None:
    """Rescale the print region until the logo's width along the surface matches the target mm."""
    if not spec.size:
        return
    ppm = spec.size["px_per_mm"]
    target = spec.size["logo_width_mm"]
    first = _mapping(spec, aspect, shape).surface_width_px / ppm
    implied = first
    pl = spec.placement
    for _ in range(4):
        k = target / max(implied, 1e-6)
        if abs(k - 1) <= 0.02:
            break
        if isinstance(spec.surface, Cylinder):
            pl.arc_deg = float(np.clip(pl.arc_deg * k, 2, 300))
            c = (pl.v_range[0] + pl.v_range[1]) / 2
            hh = (pl.v_range[1] - pl.v_range[0]) / 2 * k
            pl.v_range = (max(0.0, c - hh), min(1.0, c + hh))
        else:
            u0, v0, u1, v1 = pl.uv_box
            cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
            hu, hv = (u1 - u0) / 2 * k, (v1 - v0) / 2 * k
            pl.uv_box = (cu - hu, cv - hv, cu + hu, cv + hv)
        implied = _mapping(spec, aspect, shape).surface_width_px / ppm
    ref = spec.size["ref_what"]
    if abs(implied - target) / target > 0.04:
        notes.append(f"size: could only reach {implied:.0f}mm of the {target:.0f}mm target - the printable "
                     f"{'height of the body' if isinstance(spec.surface, Cylinder) else 'region'} limits it (reference: {ref})")
    elif abs(first - target) / target > 0.04:
        notes.append(f"size: planner box gave {first:.0f}mm; rescaled to {implied:.0f}mm (target {target:.0f}mm, reference: {ref})")
    else:
        notes.append(f"size: {implied:.0f}mm wide (target {target:.0f}mm, reference: {ref})")


def run(photo_u8: np.ndarray, logo: Logo, plan: MockupPlan, refine: bool = True, seed: int = 0) -> MockupResult:
    photo = photo_u8.astype(np.float32) / 255.0
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    cur = photo.copy()
    foot = np.zeros((h, w), np.float32)
    res = MockupResult(photo_u8.copy(), foot)
    for i, spec in enumerate(plan.placements):
        notes: list[str] = []
        lg = prepare_logo(logo, spec)
        if refine:
            if isinstance(spec.surface, Cylinder):
                spec.surface, n = refine_cylinder(gray, spec.surface, spec.placement.v_range)
                notes += n
            elif spec.snap:
                spec.surface, n = snap_quad(gray, spec.surface)
                notes += n
        enforce_size(spec, lg.aspect, (h, w), notes)
        occ = None
        if spec.occluders:
            mask = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(mask)
            for poly in spec.occluders:
                d.polygon([tuple(p) for p in poly], fill=255)
            occ = cv2.GaussianBlur(np.asarray(mask, np.float32) / 255.0, (0, 0), 1.0)
        c = composite(cur, lg.rgba, spec.surface, spec.placement, spec.look, occluder=occ, seed=seed + i)
        cur = c.rgb.astype(np.float32) / 255.0
        foot = np.maximum(foot, c.footprint)
        res.per_placement.append({"notes": notes + lg.notes, "stats": c.stats, "mapping": c.mapping, "logo": lg,
                                  "spec": spec})
    out = (np.clip(np.round(cur * 255), 0, 255)).astype(np.uint8)
    outside = foot <= 0
    out[outside] = photo_u8[outside]  # scope lock (already identical; enforced and checked)
    res.rgb = out
    res.footprint = foot
    res.identical_outside = bool(np.array_equal(out[outside], photo_u8[outside]))
    return res


def analyse(photo_u8: np.ndarray, logo: Logo, plan: MockupPlan, res: MockupResult) -> tuple[list[str], list[str]]:
    measured, possible = [], []
    h, w = photo_u8.shape[:2]
    measured.append(f"logo footprint: {(res.footprint > 0).mean():.2%} of the photo changed; "
                    f"pixels outside identical to the product photo: {'yes' if res.identical_outside else 'NO'}")
    for i, pp in enumerate(res.per_placement, 1):
        spec, m, st, lg = pp["spec"], pp["mapping"], pp["stats"], pp["logo"]
        tag = f"placement {i}"
        measured.extend(f"{tag}: {n}" for n in pp["notes"])
        fw, fh = m.footprint_px
        if not m.valid.any():
            measured.append(f"{tag}: NOTHING VISIBLE was rendered - the placement misses the visible surface")
        measured.append(f"{tag}: logo ~{fw:.0f}x{fh:.0f}px on the photo; matched softness sigma {st['blur_sigma']:.1f}px, "
                        f"grain {st['grain_sigma']:.3f}")
        if not lg.vector:
            ratio = lg.rgba.shape[1] / max(fw, 1)
            if ratio < 1:
                measured.append(f"{tag}: raster logo is upscaled {1 / ratio:.1f}x - edges will look soft/pixelated; "
                                "supply a larger PNG or an SVG")
        if m.region_valid_frac < 0.999:
            measured.append(f"{tag}: {(1 - m.region_valid_frac):.0%} of the logo box falls off the visible surface and is hidden")
        if m.max_angle_deg > 60:
            measured.append(f"{tag}: logo wraps to {m.max_angle_deg:.0f}° on the cylinder - its sides are strongly foreshortened")
        # contrast against the product
        base = np.array(st["base_srgb"], np.float32)
        vis = lg.rgba[..., 3] > 0.5
        if vis.any():
            cols = lg.rgba[vis][:, :3][:: max(1, vis.sum() // 4000)]
            de = np.linalg.norm(_lab(cols) - _lab(base[None]), axis=1)
            low = float((de < 12).mean())
            if low > 0.05:
                measured.append(f"{tag}: {low:.0%} of the logo is nearly the same colour as the product (ΔE<12) "
                                "and will be barely visible")
        if spec.size:
            from .logo import stroke_stats
            from .placements import PRODUCTION_MIN_MM
            mm = spec.size["logo_width_mm"]
            layers = stroke_stats(lg)
            line_mm = min(L["line"] for L in layers) * mm
            gaps = [L["gap"] for L in layers if L["gap"] is not None]
            gap_mm = min(gaps) * mm if gaps else None
            verdict = ", ".join(
                f"{name}: {'OK' if line_mm >= ml and (gap_mm is None or gap_mm >= mg) else 'TOO THIN'}"
                for name, (ml, mg) in PRODUCTION_MIN_MM.items())
            detail = "; ".join(f"{L['colour']} line {L['line'] * mm:.2f}mm"
                               + (f", gap {L['gap'] * mm:.2f}mm" if L["gap"] is not None else "") for L in layers)
            measured.append(f"{tag}: {len(layers)} ink colour(s) at {mm:.0f}mm wide [{detail}] -> {verdict}")
        lo, hi = st.get("shade_range", [1, 1])
        if lo < 0.45:
            possible.append(f"{tag}: deep shadows/folds cross the logo (shading down to {lo:.0%}); parts will look dark - this is how the real print would look")
        if spec.occluders:
            possible.append(f"{tag}: occluder outlines were estimated from the preview; their edges can be a few px off")
        if spec.surface.kind == "flat" and spec.surface.fabric:
            possible.append(f"{tag}: fabric folds are inferred from shading; very strong creases may bend the print less than in reality")
        if isinstance(spec.surface, Cylinder) or spec.snap is False:
            possible.append(f"{tag}: surface geometry was estimated from a downscaled preview; check the guide overlay (--guide)")
    if not res.identical_outside:  # pragma: no cover
        measured.append("ERROR: scope verification failed - do not use this output")
    return measured, possible


def guide_image(photo_u8: np.ndarray, plan: MockupPlan, res: MockupResult) -> Image.Image:
    im = Image.fromarray(res.rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(1, round(min(im.size) / 500))
    for pp in res.per_placement:
        spec, lg = pp["spec"], pp["logo"]
        for line in guide_lines(spec.surface, spec.placement, lg.aspect, photo_u8.shape[:2]):
            d.line([tuple(map(float, p)) for p in line], fill=(0, 200, 255), width=lw)
    return im


def load_photo(path: str):
    src = base_engine.load(path)
    return src, src.u8
