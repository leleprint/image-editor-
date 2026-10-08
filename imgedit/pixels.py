"""Low-level float pixel helpers. All images are float32 arrays in [0, 1]."""

from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709


def to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue in degrees [0, 360), saturation and value in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    safe = np.where(d == 0, 1.0, d)
    h = np.where(mx == r, (g - b) / safe % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = np.where(d == 0, 0.0, h * 60.0)
    s = np.where(mx == 0, 0.0, d / np.where(mx == 0, 1.0, mx))
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 360.0) / 60.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    choices = [
        np.stack([v, t, p], -1), np.stack([q, v, p], -1), np.stack([p, v, t], -1),
        np.stack([p, q, v], -1), np.stack([t, p, v], -1), np.stack([v, p, q], -1),
    ]
    out = np.zeros(h.shape + (3,), dtype=np.float32)
    for k, c in enumerate(choices):
        out = np.where((i == k)[..., None], c, out)
    return out


def _box_blur_axis(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="reflect" if a.shape[axis] > r + 1 else "edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian approximation by three box passes (O(1) per pixel in sigma)."""
    if sigma <= 0.3:
        return a.astype(np.float32, copy=True)
    # Box sizes for 3 passes approximating the given sigma (Kovesi).
    n = 3
    w_ideal = np.sqrt(12 * sigma * sigma / n + 1)
    wl = int(np.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m = round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4))
    sizes = [wl if i < m else wu for i in range(n)]
    out = a.astype(np.float32)
    for s in sizes:
        r = (s - 1) // 2
        out = _box_blur_axis(out, r, 0)
        out = _box_blur_axis(out, r, 1)
    return out


def parse_color(value: str) -> tuple[float, float, float]:
    named = {
        "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00ff00",
        "blue": "#0000ff", "yellow": "#ffff00", "gray": "#808080", "grey": "#808080",
    }
    v = named.get(str(value).strip().lower(), str(value).strip())
    if not v.startswith("#") or len(v) not in (4, 7):
        raise ValueError(f"invalid color {value!r}; use #rrggbb")
    if len(v) == 4:
        v = "#" + "".join(ch * 2 for ch in v[1:])
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[return-value]
