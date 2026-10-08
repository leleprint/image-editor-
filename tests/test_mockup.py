import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from imgedit.cli import main
from imgedit.mockup import engine as mk
from imgedit.mockup.geometry import Cylinder, Flat, Placement, map_cylinder, map_flat
from imgedit.mockup.logo import Logo, knockout_white, load_logo
from imgedit.mockup.plan import MOCKUP_SCHEMA, validate_mockup
from imgedit.mockup.planner import MOCKUP_SYSTEM, plan_mockup
from imgedit.mockup.refine import refine_cylinder, snap_quad
from imgedit.mockup.render import Look, composite


def logo_rgba(w=200, h=100, color=(0.9, 0.1, 0.1)):
    a = np.zeros((h, w, 4), np.float32)
    a[10:-10, 10:-10, :3] = color
    a[10:-10, 10:-10, 3] = 1
    return a


def box_scene(W=800, H=600):
    img = np.full((H, W, 3), 0.85, np.float32)
    face = np.array([[200, 150], [600, 170], [590, 480], [210, 500]], np.float32)
    cv2.fillPoly(img, [face.astype(np.int32)], (0.7, 0.55, 0.38))
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return (img * 255).astype(np.uint8), face


def mug_scene(W=800, H=700, cx=400.0, R=150.0):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W, 3), 0.85, np.float32)
    body = (np.abs(x - cx) < R) & (y > 150) & (y < 550)
    s = np.clip((x - cx) / R, -1, 1)
    shade = 0.25 + 0.7 * np.sqrt(np.clip(1 - s * s, 0, 1))
    img[body] = shade[body][:, None] * np.array([0.97, 0.96, 0.95])
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return (img * 255).astype(np.uint8)


def raw_plan(placement, clauses=("put the logo on it",)):
    return {"understanding": "", "clauses": list(clauses), "ambiguities": [], "needs_clarification": False,
            "question": "", "product": {"type": "x", "color": "x", "material": "other", "finish": "matte"},
            "placements": [placement], "unsupported": [], "preserved": [], "risks": []}


def flat_placement(quad, snap=False, kind="flat", size=None, uv=(0.2, 0.2, 0.8, 0.8)):
    return {"clause": 0, "standard": "custom", "why": "", "surface": {"kind": kind, "quad": quad, "snap": snap, "cylinder": None},
            "region": {"uv_box": list(uv), "center_deg": 0, "arc_deg": 60, "v_range": [0, 1], "anchor": "center"},
            "size": size, "look": {"technique": "print", "opacity": 0.97, "gloss": 0, "texture": 0, "ink": "original",
                                   "ink_color": ""}, "occluders": []}


# ------------------------------------------------------------ geometry

def test_flat_mapping_hits_quad_and_keeps_aspect():
    q = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)
    m = map_flat(Flat(q), Placement(), aspect=2.0, shape=(400, 400), ss=1)
    ys, xs = np.nonzero(m.valid)
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    assert abs(w / h - 2.0) < 0.05
    assert abs((xs.min() + m.roi[0]) - 100) <= 1 and abs((xs.max() + m.roi[0]) - 299) <= 1


def test_cylinder_mapping_symmetric_and_foreshortened():
    c = Cylinder(np.array([400.0, 100]), np.array([400.0, 500]), 150, 150, 20, 20)
    m = map_cylinder(c, Placement(center_deg=0, arc_deg=120, v_range=(0.2, 0.8)), 1.0, (600, 800), 1)
    ys, xs = np.nonzero(m.valid)
    xs = xs + m.roi[0]
    assert abs((xs.min() + xs.max()) / 2 - 400) <= 2
    # equal logo-u steps are narrower near the silhouette than in the middle
    row = m.lu[int(np.median(ys))]
    cols = np.nonzero(m.valid[int(np.median(ys))])[0]
    du = np.diff(row[cols])
    assert du[:5].mean() > du[len(du) // 2 - 2:len(du) // 2 + 3].mean()


# ------------------------------------------------------------ validation

def test_validator_orders_quad_and_rejects_bad():
    p = validate_mockup(raw_plan(flat_placement([[0.6, 0.6], [0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])), (100, 100))
    q = p.placements[0].surface.quad
    assert np.allclose(q[0], [20, 20]) and np.allclose(q[2], [60, 60])
    bad = validate_mockup(raw_plan(flat_placement([[0.2, 0.2], [0.6, 0.6], [0.6, 0.2], [0.2, 0.6]]) | {"clause": 5}),
                          (100, 100))
    assert not bad.placements and any("not part of the request" in i for i in bad.issues)


def test_validator_flags_size_outside_standard():
    pl = flat_placement([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], kind="fabric",
                        size={"logo_width_mm": 300, "ref_p0": [0, 0.5], "ref_p1": [1, 0.5], "ref_length_mm": 500,
                              "ref_what": "chest"})
    pl["standard"] = "tshirt.left_chest"
    p = validate_mockup(raw_plan(pl), (1000, 1000))
    assert any("outside the usual" in i for i in p.issues)


def test_schema_mentions_all_kinds():
    assert MOCKUP_SCHEMA["properties"]["placements"]["items"]["properties"]["surface"]["properties"]["kind"]["enum"] == \
        ["flat", "fabric", "cylinder"]
    assert "tshirt.left_chest" in MOCKUP_SYSTEM


# ------------------------------------------------------------ refinement

def test_edge_fit_snaps_box_face_subpixel():
    img, face = box_scene()
    gray = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    noisy = face + np.array([[8, -7], [-9, 6], [7, 8], [-6, -9]], np.float32)
    s, notes = snap_quad(gray, Flat(noisy))
    assert np.hypot(*(s.quad - face).T).max() < 2.0, notes


def test_cylinder_refine_finds_silhouette():
    img = mug_scene()
    gray = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    est = Cylinder(np.array([412.0, 160]), np.array([412.0, 540]), 140, 140, 15, 15)
    s, notes = refine_cylinder(gray, est, (0.1, 0.9))
    assert abs(s.top[0] - 400) < 1.5 and abs(s.r_top - 150) < 1.5, notes


# ------------------------------------------------------------ rendering

def test_composite_scope_lock_and_shading():
    img = mug_scene()
    c = Cylinder(np.array([400.0, 160]), np.array([400.0, 540]), 150, 150, 15, 15)
    out = composite(img.astype(np.float32) / 255, logo_rgba(color=(1, 1, 1)), c,
                    Placement(center_deg=0, arc_deg=140, v_range=(0.3, 0.7)), Look(gloss=0, grain=0))
    outside = out.footprint <= 0
    assert np.array_equal(out.rgb[outside], img[outside])
    # a white logo is darker toward the silhouette, following the mug's own shading
    ys, xs = np.nonzero(out.footprint > 0.99)
    row = int(np.median(ys))
    cols = xs[ys == row]
    assert out.rgb[row, cols.min() + 2].mean() < out.rgb[row, 400].mean() - 20


def test_run_enforces_physical_size_flat():
    img, face = box_scene()
    pl = flat_placement((face / [800, 600]).tolist(), snap=True,
                        size={"logo_width_mm": 100, "ref_p0": [200 / 800, 0.3], "ref_p1": [600 / 800, 0.3],
                              "ref_length_mm": 400, "ref_what": "box width"})
    plan = validate_mockup(raw_plan(pl), (800, 600))
    logo = Logo("x.png", logo_rgba(), vector=False)
    res = mk.run(img, logo, plan)
    assert res.identical_outside
    w = res.per_placement[0]["mapping"].surface_width_px
    assert abs(w - 100) < 5  # 1 px/mm reference
    assert any("rescaled" in n for n in res.per_placement[0]["notes"])


def test_run_enforces_physical_size_cylinder():
    img = mug_scene()
    pl = {"clause": 0, "standard": "mug11.front", "why": "",
          "surface": {"kind": "cylinder", "quad": None, "snap": False,
                      "cylinder": {"top": [0.5, 160 / 700], "bottom": [0.5, 540 / 700], "r_top": 150 / 800,
                                   "r_bottom": 150 / 800, "b_top": 15 / 700, "b_bottom": 15 / 700}},
          "region": {"uv_box": [0, 0, 1, 1], "center_deg": 0, "arc_deg": 40, "v_range": [0.1, 0.9], "anchor": "center"},
          "size": {"logo_width_mm": 80, "ref_p0": [250 / 800, 0.5], "ref_p1": [550 / 800, 0.5], "ref_length_mm": 82,
                   "ref_what": "diameter"},
          "look": {"technique": "print", "opacity": 0.98, "gloss": 0.8, "texture": 0, "ink": "original", "ink_color": ""},
          "occluders": []}
    plan = validate_mockup(raw_plan(pl), (800, 700))
    res = mk.run(img, Logo("x.png", logo_rgba(), False), plan)
    arc_mm = res.per_placement[0]["mapping"].surface_width_px / (300 / 82)
    assert abs(arc_mm - 80) < 3


def test_analysis_flags_low_contrast_and_upscaled_raster():
    img = mug_scene()
    c = {"top": [0.5, 160 / 700], "bottom": [0.5, 540 / 700], "r_top": 150 / 800, "r_bottom": 150 / 800,
         "b_top": 0.02, "b_bottom": 0.02}
    pl = {"clause": 0, "standard": "custom", "why": "", "surface": {"kind": "cylinder", "quad": None, "snap": False,
                                                                     "cylinder": c},
          "region": {"uv_box": [0, 0, 1, 1], "center_deg": 0, "arc_deg": 120, "v_range": [0.2, 0.8], "anchor": "center"},
          "size": None, "look": {"technique": "print", "opacity": 0.98, "gloss": 0, "texture": 0, "ink": "original",
                                 "ink_color": ""}, "occluders": []}
    plan = validate_mockup(raw_plan(pl), (800, 700))
    small = Logo("x.png", logo_rgba(60, 40, color=(0.96, 0.95, 0.94)), False)
    res = mk.run(img, small, plan)
    measured, _ = mk.analyse(img, small, plan, res)
    text = "\n".join(measured)
    assert "barely visible" in text and "upscaled" in text


def test_knockout_white():
    a = np.ones((50, 50, 4), np.float32)
    a[20:30, 20:30, :3] = (0.1, 0.2, 0.8)
    lg = knockout_white(Logo("x.png", a, False))
    assert lg.rgba.shape[0] <= 16  # trimmed to the artwork
    assert lg.rgba[..., 3].max() > 0.9 and lg.rgba[0, 0, 3] == 0


def test_svg_logo_loads(tmp_path):
    p = tmp_path / "l.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100"><rect x="10" y="10" width="180" height="80" fill="#c00"/></svg>')
    lg = load_logo(str(p), target_width=800)
    assert lg.vector and abs(lg.aspect - 180 / 80) < 0.05


# ------------------------------------------------------------ planner + CLI

def test_planner_sends_both_images_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEDIT_CACHE", str(tmp_path / "c"))
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{}")],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    plan_mockup(b"jpg", b"png", "facts", "left chest", client=client)
    plan_mockup(b"jpg", b"png", "facts", "left chest", client=client)
    assert len(calls) == 1
    content = calls[0]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert calls[0]["output_config"]["effort"] == "high"


def test_cli_mockup_apply(tmp_path):
    img, face = box_scene()
    prod = tmp_path / "box.jpg"
    Image.fromarray(img).save(prod, quality=95)
    logo = tmp_path / "logo.png"
    Image.fromarray((logo_rgba() * 255).astype(np.uint8), "RGBA").save(logo)
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps(raw_plan(flat_placement((face / [800, 600]).tolist(), snap=True))))
    out = tmp_path / "o.png"
    assert main(["mockup-apply", str(prod), str(logo), str(plan), "-o", str(out), "--yes", "--guide"]) == 0
    assert out.exists() and (tmp_path / "o.guide.png").exists()
    assert main(["mockup-apply", str(prod), str(logo), str(plan), "-o", str(prod), "--yes"]) == 1


def _ring_logo(n=1200, width_frac=0.006):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - n / 2, y - n / 2)
    a = np.zeros((n, n, 4), np.float32)
    a[..., :3] = 1
    a[..., 3] = (np.abs(r - n * 0.45) < n * width_frac / 2).astype(np.float32)
    return a


def test_thin_white_line_on_black_is_continuous_and_linear_light():
    """Regression: sub-pixel white strokes on dark products must not break up (aliasing) or go dark (sRGB blending)."""
    photo = np.full((120, 120, 3), 0.04, np.float32)
    q = np.array([[40, 40], [80, 40], [80, 80], [40, 80]], float)
    c = composite(photo, _ring_logo(), Flat(q, fabric=False), Placement(), Look(blur=0, grain=0, opacity=1.0))
    lum = c.rgb.astype(np.float32).mean(-1)
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    xs = (60 + 18 * np.cos(t)).round().astype(int)
    ys = (60 + 18 * np.sin(t)).round().astype(int)
    on_ring = np.array([lum[yy - 1:yy + 2, xx - 1:xx + 2].max() for xx, yy in zip(xs, ys)])
    assert on_ring.min() > 25  # no gaps anywhere around the ring
    # ~0.24px coverage of white over sRGB 10 must read clearly above the background (linear-light averaging)
    assert np.median(on_ring) > 60


def test_dark_product_gets_no_dye_tint():
    photo = np.full((120, 120, 3), (0.06, 0.075, 0.055), np.float32)  # greenish black fabric
    q = np.array([[30, 30], [90, 30], [90, 90], [30, 90]], float)
    logo = np.ones((50, 50, 4), np.float32)
    c = composite(photo, logo, Flat(q), Placement(), Look(blur=0, grain=0, opacity=1.0))
    px = c.rgb[60, 60].astype(int)
    assert px.max() - px.min() <= 3  # white ink stays neutral white
