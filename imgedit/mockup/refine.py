"""Sub-pixel refinement of planner-estimated geometry against real image edges.

The planner sees a downscaled preview, so its coordinates are good to a few
percent. These refiners snap them to measurable features and report how far
they moved; if no reliable feature is found the estimate is kept and the
report says so.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Cylinder, Flat


def _fit_edges(gray: np.ndarray, q: np.ndarray, r: int) -> np.ndarray | None:
    """Fit a robust line to each face edge (strongest gradient along the normal), intersect neighbours."""
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        pts = []
        for t in np.linspace(0.12, 0.88, 40):  # avoid the corners themselves (junctions are ambiguous)
            c = a + d * t
            offs = np.arange(-r, r + 1, 0.5)
            ps = c[None] + offs[:, None] * n[None]
            xi = np.clip(ps[:, 0].round().astype(int), 0, w - 1)
            yi = np.clip(ps[:, 1].round().astype(int), 0, h - 1)
            mag = np.abs(gx[yi, xi] * n[0] + gy[yi, xi] * n[1])
            k = int(np.argmax(mag))
            if mag[k] > 3 * (np.median(mag) + 1e-4) and mag[k] > 0.08:
                if 0 < k < len(offs) - 1:  # parabolic sub-sample peak
                    y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
                    den = y0 - 2 * y1 + y2
                    k = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0)
                pts.append(c + (offs[0] + k * 0.5) * n)
        if len(pts) < 12:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = np.zeros((4, 2), np.float32)
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A, p2 - p1)
        out[i] = p1 + t[0] * d1
    return out


def snap_quad(gray: np.ndarray, s: Flat, radius_frac: float = 0.025) -> tuple[Flat, list[str]]:
    h, w = gray.shape
    r = int(max(6, radius_frac * np.hypot(h, w)))
    q0 = np.asarray(s.quad, np.float32)
    fitted = _fit_edges(gray, q0, r)
    if fitted is not None:
        moved = np.hypot(*(fitted - q0).T)
        if moved.max() <= 1.5 * r and cv2.isContourConvex(fitted.reshape(-1, 1, 2)):
            return Flat(fitted, s.fabric, s.warp_strength), [
                f"edge fit: 4 face edges located; corners moved {', '.join(f'{m:.1f}' for m in moved)} px"]

    h, w = gray.shape
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    q = np.asarray(s.quad, np.float32).copy()
    notes, moved = ["edge fit failed; fell back to corner detection"], []
    for i, (x, y) in enumerate(q):
        x0, y0 = int(max(0, x - r)), int(max(0, y - r))
        x1, y1 = int(min(w, x + r)), int(min(h, y + r))
        if x1 - x0 < 8 or y1 - y0 < 8:
            moved.append(None)
            continue
        pts = cv2.goodFeaturesToTrack(g8[y0:y1, x0:x1], maxCorners=8, qualityLevel=0.08, minDistance=3)
        if pts is None:
            moved.append(None)
            continue
        pts = pts.reshape(-1, 2) + [x0, y0]
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        best = pts[np.argmin(d)].astype(np.float32).reshape(1, 1, 2)
        cv2.cornerSubPix(g8, best, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        q[i] = best.reshape(2)
        moved.append(float(np.hypot(*(q[i] - [x, y]))))
    snapped = [m for m in moved if m is not None]
    if len(snapped) < 4:
        notes.append(f"corner snap: {len(snapped)}/4 corners found a real image corner; others keep the planner estimate")
    if snapped:
        notes.append(f"corner snap: moved corners by {', '.join(f'{m:.1f}' for m in snapped)} px")
    # reject a snap that collapses the quad
    area = cv2.contourArea(q.reshape(-1, 1, 2))
    if area < 0.5 * cv2.contourArea(np.asarray(s.quad, np.float32).reshape(-1, 1, 2)):
        return s, ["corner snap rejected (would distort the face); planner estimate kept"]
    return Flat(q, s.fabric, s.warp_strength), notes


def refine_cylinder(gray: np.ndarray, s: Cylinder, v_range: tuple[float, float]) -> tuple[Cylinder, list[str]]:
    """Find the left/right silhouette edges on rows across the print band; refit axis and radii."""
    h, w = gray.shape
    top, bot = np.asarray(s.top, float), np.asarray(s.bottom, float)
    axis = bot - top
    L = np.linalg.norm(axis)
    ay = axis / L
    ax = np.array([-ay[1], ay[0]])
    if ax[0] < 0:
        ax = -ax
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    samples = []
    for v in np.linspace(max(0.05, v_range[0]), min(0.95, v_range[1]), 9):
        r = s.r_top + v * (s.r_bottom - s.r_top)
        c = top + ay * v * L
        found = []
        for side in (-1, 1):
            ts = np.linspace(r * 0.85, r * 1.15, max(61, int(r * 0.3 * 4)))
            pts = c[None] + side * ts[:, None] * ax[None]
            mag = np.abs(cv2.remap(gx, pts[:, 0:1].astype(np.float32), pts[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel() * ax[0]
                         + cv2.remap(gy, pts[:, 0:1].astype(np.float32), pts[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel() * ax[1])
            k = int(np.argmax(mag))
            ok = mag[k] > 4 * (np.median(mag) + 1e-3) and mag[k] > 0.15
            kk = float(k)
            if ok and 0 < k < len(ts) - 1:
                y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
                den = y0 - 2 * y1 + y2
                kk = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
            found.append(side * float(np.interp(kk, np.arange(len(ts)), ts)) if ok else None)
        if None not in found:
            centre_off = (found[0] + found[1]) / 2
            radius = (found[1] - found[0]) / 2
            samples.append((v, centre_off, radius))
    if len(samples) < 4:
        return s, ["cylinder refine: silhouette edges not found reliably; planner estimate kept"]
    vs, offs, rads = map(np.array, zip(*samples))
    po = np.polyfit(vs, offs, 1)
    pr = np.polyfit(vs, rads, 1)
    resid = np.abs(np.polyval(pr, vs) - rads)
    if resid.max() > 0.05 * rads.mean():
        return s, ["cylinder refine: edges inconsistent (handle/occlusion?); planner estimate kept"]
    new_top = top + ax * po[1]
    new_bot = bot + ax * (po[0] + po[1])
    nt, nb = float(pr[1]), float(pr[0] + pr[1])
    ratio_t = s.b_top / max(s.r_top, 1e-6)
    ratio_b = s.b_bottom / max(s.r_bottom, 1e-6)
    out = Cylinder(new_top, new_bot, nt, nb, ratio_t * nt, ratio_b * nb)
    return out, [f"cylinder refine: axis moved {abs(po[1]):.1f}px, radius {s.r_top:.0f}->{nt:.0f}px (top), "
                 f"{s.r_bottom:.0f}->{nb:.0f}px (bottom) from {len(samples)} edge rows"]
