"""Surface models that map product-photo pixels back to logo coordinates.

Every model is an *inverse* map: for each output pixel it returns the logo
coordinate (lu, lv) in [0,1]^2 that lands there, plus a validity mask. That
lets the renderer sample the logo once, supersampled, with no holes.

  flat     - rigid planar face (box, card, sign, screen, flat bag): homography
  fabric   - planar homography + displacement along folds (shirts, totes, caps)
  cylinder - straight or tapered cylinder (mugs, bottles, cans, pens), any tilt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Placement:
    """Where the logo sits inside the surface's print region."""

    uv_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # flat/fabric print region in face UV
    center_deg: float = 0.0  # cylinder: logo centre angle, 0 = facing camera, + = to the right
    arc_deg: float = 60.0  # cylinder: print region angular width
    v_range: tuple[float, float] = (0.2, 0.8)  # cylinder: print region along axis, 0 = top ellipse
    anchor: str = "center"  # center | top: where the aspect-fitted logo sits inside the region


@dataclass
class Flat:
    quad: np.ndarray  # 4x2 px: TL, TR, BR, BL of the face (or print area)
    fabric: bool = False
    warp_strength: float = 1.0  # fabric only: displacement multiplier

    kind = "flat"


@dataclass
class Cylinder:
    top: np.ndarray  # axis point at the top ellipse centre (px)
    bottom: np.ndarray  # axis point at the bottom ellipse centre (px)
    r_top: float  # horizontal radius at top (px)
    r_bottom: float
    b_top: float  # ellipse minor half-axis at top (px); >0 = seen from above
    b_bottom: float

    kind = "cylinder"


@dataclass
class Mapping:
    roi: tuple[int, int, int, int]  # x0, y0, x1, y1 in output px
    lu: np.ndarray  # logo u per ROI pixel (at supersampled resolution)
    lv: np.ndarray
    valid: np.ndarray
    ss: int  # supersampling factor
    region_valid_frac: float  # fraction of the fitted logo box that is visible
    footprint_px: tuple[float, float]  # approx logo size on the photo (w, h) px
    max_angle_deg: float = 0.0  # cylinder: steepest visible angle covered by the logo
    surface_width_px: float = 0.0  # logo width measured ALONG the surface (arc length on cylinders)
    notes: list[str] = field(default_factory=list)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def _fit_aspect(rw: float, rh: float, aspect: float, anchor: str) -> tuple[float, float, float, float]:
    """Return logo sub-box (u0, v0, u1, v1) inside a unit region of physical size rw x rh."""
    if rw / rh > aspect:
        fw = rh * aspect / rw
        u0, u1 = 0.5 - fw / 2, 0.5 + fw / 2
        v0, v1 = 0.0, 1.0
    else:
        fh = rw / aspect / rh
        u0, u1 = 0.0, 1.0
        v0, v1 = (0.0, fh) if anchor == "top" else (0.5 - fh / 2, 0.5 + fh / 2)
    return u0, v0, u1, v1


def _roi(points: np.ndarray, shape: tuple[int, int], pad: int) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0 = np.floor(points.min(0)).astype(int) - pad
    x1, y1 = np.ceil(points.max(0)).astype(int) + pad
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _grid(roi, ss):
    x0, y0, x1, y1 = roi
    xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss
    ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss
    return np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))


def fold_displacement(photo_lum: np.ndarray, scale_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Displacement field from fabric folds: ink follows the surface slope (luminance proxy)."""
    sigma = max(1.5, scale_px * 0.012)
    sm = cv2.GaussianBlur(photo_lum, (0, 0), sigma)
    base = cv2.GaussianBlur(photo_lum, (0, 0), sigma * 8)
    relief = sm - base
    gy, gx = np.gradient(relief)
    amp = scale_px * 0.35
    return (gx * amp).astype(np.float32), (gy * amp).astype(np.float32)


def map_flat(s: Flat, pl: Placement, aspect: float, shape: tuple[int, int], ss: int,
             disp: Optional[tuple[np.ndarray, np.ndarray]] = None) -> Mapping:
    q = np.asarray(s.quad, np.float64)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
    H = _homography(unit, q)
    face_w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    face_h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    u0, v0, u1, v1 = pl.uv_box
    su0, sv0, su1, sv1 = _fit_aspect(face_w * (u1 - u0), face_h * (v1 - v0), aspect, pl.anchor)
    box = np.array([[u0 + su0 * (u1 - u0), v0 + sv0 * (v1 - v0)], [u0 + su1 * (u1 - u0), v0 + sv0 * (v1 - v0)],
                    [u0 + su1 * (u1 - u0), v0 + sv1 * (v1 - v0)], [u0 + su0 * (u1 - u0), v0 + sv1 * (v1 - v0)]])
    corners = cv2.perspectiveTransform(box[None].astype(np.float64), H)[0]
    Hl = _homography(unit, corners)  # logo unit square -> photo
    Hinv = np.linalg.inv(Hl)
    pad = 4 + (int(face_w * 0.03) if s.fabric else 0)
    roi = _roi(corners, shape, pad)
    X, Y = _grid(roi, ss)
    if disp is not None:
        dx, dy = disp
        x0, y0, x1, y1 = roi
        dxr = cv2.resize(dx[y0:y1, x0:x1], (X.shape[1], X.shape[0]), interpolation=cv2.INTER_LINEAR)
        dyr = cv2.resize(dy[y0:y1, x0:x1], (X.shape[1], X.shape[0]), interpolation=cv2.INTER_LINEAR)
        X = X - dxr * s.warp_strength
        Y = Y - dyr * s.warp_strength
    den = Hinv[2, 0] * X + Hinv[2, 1] * Y + Hinv[2, 2]
    lu = (Hinv[0, 0] * X + Hinv[0, 1] * Y + Hinv[0, 2]) / den
    lv = (Hinv[1, 0] * X + Hinv[1, 1] * Y + Hinv[1, 2]) / den
    valid = (den > 0) & (lu >= 0) & (lu <= 1) & (lv >= 0) & (lv <= 1)
    fw = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
    fh = (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])) / 2
    # visible fraction: corners inside the image
    h, w = shape
    inside = np.mean([(0 <= c[0] <= w) and (0 <= c[1] <= h) for c in corners])
    return Mapping(roi, lu.astype(np.float32), lv.astype(np.float32), valid, ss, float(inside), (fw, fh),
                   surface_width_px=float(fw))


def map_cylinder(s: Cylinder, pl: Placement, aspect: float, shape: tuple[int, int], ss: int) -> Mapping:
    top, bot = np.asarray(s.top, np.float64), np.asarray(s.bottom, np.float64)
    axis = bot - top
    L = float(np.linalg.norm(axis))
    ay = axis / L
    ax = np.array([ay[1], -ay[0]]) * -1  # unit vector to the image-right of the axis
    if ax[0] < 0:
        ax = -ax
    r_mean = (s.r_top + s.r_bottom) / 2
    b_mean = (s.b_top + s.b_bottom) / 2
    elev = np.arcsin(np.clip(abs(b_mean) / max(r_mean, 1e-6), 0, 0.95))
    v0, v1 = pl.v_range
    arc = np.radians(pl.arc_deg)
    region_w = arc * r_mean
    region_h = (v1 - v0) * L / max(np.cos(elev), 0.3)
    su0, sv0, su1, sv1 = _fit_aspect(region_w, region_h, aspect, pl.anchor)
    th0 = np.radians(pl.center_deg) - arc / 2 + su0 * arc
    th1 = np.radians(pl.center_deg) - arc / 2 + su1 * arc
    lv0, lv1 = v0 + sv0 * (v1 - v0), v0 + sv1 * (v1 - v0)

    # ROI from sampled forward points of the logo box
    ts = np.linspace(th0, th1, 25)
    vs = np.linspace(lv0, lv1, 25)
    T, V = np.meshgrid(ts, vs)
    R = s.r_top + V * (s.r_bottom - s.r_top)
    B = s.b_top + V * (s.b_bottom - s.b_top)
    vis = np.cos(T) > 0
    px = top[0] + ax[0] * R * np.sin(T) + ay[0] * (V * L + B * np.cos(T))
    py = top[1] + ax[1] * R * np.sin(T) + ay[1] * (V * L + B * np.cos(T))
    pts = np.stack([px[vis], py[vis]], -1) if vis.any() else np.array([[top[0], top[1]]])
    roi = _roi(pts, shape, 4)
    X, Y = _grid(roi, ss)
    dX, dY = X - top[0], Y - top[1]
    qx = dX * ax[0] + dY * ax[1]
    qy = dX * ay[0] + dY * ay[1]
    v = qy / L
    sin_t = np.zeros_like(v)
    cos_t = np.ones_like(v)
    for _ in range(4):
        r = s.r_top + v * (s.r_bottom - s.r_top)
        sin_t = np.clip(qx / np.maximum(r, 1e-6), -1.5, 1.5)
        cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
        v = (qy - s.b_top * cos_t) / (L + (s.b_bottom - s.b_top) * cos_t)
    theta = np.arctan2(sin_t, cos_t)
    lu = (theta - th0) / (th1 - th0)
    lv = (v - lv0) / (lv1 - lv0)
    valid = (np.abs(sin_t) < 1) & (lu >= 0) & (lu <= 1) & (lv >= 0) & (lv <= 1)
    vis_frac = float(np.mean(np.cos(np.linspace(th0, th1, 200)) > 0.05))
    max_ang = float(np.degrees(max(abs(th0), abs(th1))))
    fw = r_mean * (np.sin(min(th1, np.pi / 2)) - np.sin(max(th0, -np.pi / 2)))
    fh = (lv1 - lv0) * L
    return Mapping(roi, lu.astype(np.float32), lv.astype(np.float32), valid, ss, vis_frac, (float(fw), float(fh)),
                   max_angle_deg=max_ang, surface_width_px=float((th1 - th0) * r_mean))


def guide_lines(surface, pl: Placement, aspect: float, shape) -> list[np.ndarray]:
    """Polylines (px) of the logo box and a grid, for the placement-check overlay."""
    lines = []
    if surface.kind == "flat":
        m = map_flat(surface, pl, aspect, shape, 1)
        q = np.asarray(surface.quad, np.float64)
        lines.append(np.vstack([q, q[:1]]))
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
        H = _homography(unit, q)
        u0, v0, u1, v1 = pl.uv_box
        box = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]], np.float64)
        lines.append(cv2.perspectiveTransform(box[None], H)[0])
        _ = m
    else:
        s = surface
        top, bot = np.asarray(s.top, float), np.asarray(s.bottom, float)
        axis = bot - top
        L = np.linalg.norm(axis)
        ay = axis / L
        ax = np.array([ay[1], -ay[0]]) * -1
        if ax[0] < 0:
            ax = -ax
        for vv in np.linspace(0, 1, 5).tolist() + list(pl.v_range):
            t = np.linspace(-np.pi / 2, np.pi / 2, 60)
            r = s.r_top + vv * (s.r_bottom - s.r_top)
            b = s.b_top + vv * (s.b_bottom - s.b_top)
            lines.append(np.stack([top[0] + ax[0] * r * np.sin(t) + ay[0] * (vv * L + b * np.cos(t)),
                                   top[1] + ax[1] * r * np.sin(t) + ay[1] * (vv * L + b * np.cos(t))], -1))
        for tdeg in (pl.center_deg - pl.arc_deg / 2, pl.center_deg, pl.center_deg + pl.arc_deg / 2):
            t = np.radians(tdeg)
            vv = np.linspace(0, 1, 30)
            r = s.r_top + vv * (s.r_bottom - s.r_top)
            b = s.b_top + vv * (s.b_bottom - s.b_top)
            lines.append(np.stack([top[0] + ax[0] * r * np.sin(t) + ay[0] * (vv * L + b * np.cos(t)),
                                   top[1] + ax[1] * r * np.sin(t) + ay[1] * (vv * L + b * np.cos(t))], -1))
    return lines
