import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imgedit import engine
from imgedit.cli import main
from imgedit.ops import OPS, catalog_text
from imgedit.plan import PLAN_SCHEMA, validate
from imgedit.planner_llm import SYSTEM_PROMPT, plan_with_model
from imgedit.planner_local import plan_locally


@pytest.fixture
def img(tmp_path):
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x / w * 255, y / h * 200, np.full_like(x, 120)], -1).astype(np.uint8)
    a[80:110, 20:60] = [30, 160, 40]
    p = tmp_path / "in.png"
    Image.fromarray(a).save(p)
    return str(p), a


def region(shape="rect", box=(0.25, 0.25, 0.75, 0.75), feather=0.0, invert=False, color_range=None, points=None):
    return {"shape": shape, "box": list(box) if box else None, "points": points, "feather": feather,
            "invert": invert, "color_range": color_range}


def raw_plan(*steps, clauses=("do it",), **kw):
    d = {"understanding": "", "clauses": list(clauses), "ambiguities": [], "needs_clarification": False,
         "question": "", "steps": list(steps), "unsupported": [], "preserved": [], "risks": []}
    d.update(kw)
    return d


def step(op, region_=None, clause=0, **params):
    return {"clause": clause, "op": op, "params": [{"name": k, "value": v} for k, v in params.items()],
            "region": region_, "why": ""}


# ------------------------------------------------------------ scope lock

@pytest.mark.parametrize("op,params", [
    ("brightness", {"amount": 0.8}), ("contrast", {"amount": 1.0}), ("saturation", {"amount": 1.0}),
    ("blur", {"radius": 8}), ("sharpen", {"amount": 2.0, "radius": 3}), ("hue_shift", {"degrees": 90}),
    ("exposure", {"stops": 2}), ("denoise", {"strength": 1}), ("fill", {"color": "#ff00ff"}),
    ("pixelate", {"block": 8}), ("vignette", {"amount": -1}), ("levels", {"black": 0.2, "white": 0.7}),
])
@pytest.mark.parametrize("feather", [0.0, 0.05])
def test_pixels_outside_region_are_byte_identical(img, op, params, feather):
    path, a = img
    src = engine.load(path)
    plan = validate(raw_plan(step(op, region(feather=feather), **params)))
    assert not [i for i in plan.issues if "REJECTED" in i]
    res = engine.execute(src, plan)
    assert res.untouched_identical
    outside = res.touched == 0
    assert outside.mean() > 0.4
    assert np.array_equal(res.rgb[outside], a[outside])
    assert (res.rgb[~outside] != a[~outside]).any()


def test_color_range_limits_edit_to_matching_hue(img):
    path, a = img
    src = engine.load(path)
    green = {"hue_center": 125, "hue_tolerance": 20, "min_saturation": 0.3, "lum_min": 0, "lum_max": 1}
    plan = validate(raw_plan(step("brightness", region("full", None, color_range=green), amount=0.8)))
    res = engine.execute(src, plan)
    changed = (res.rgb != a).any(-1)
    assert changed[85:105, 25:55].all()
    assert changed.mean() < 0.15


def test_text_overlay_touches_only_its_glyphs(img):
    path, a = img
    src = engine.load(path)
    plan = validate(raw_plan(step("text", region("full", None), text="Hi", x=0.5, y=0.5, size=0.2)))
    res = engine.execute(src, plan)
    changed = (res.rgb != a).any(-1)
    assert 0 < changed.mean() < 0.1
    assert res.untouched_identical


def test_noop_plan_is_identity(img):
    path, a = img
    res = engine.execute(engine.load(path), validate(raw_plan(step("brightness", amount=0.0))))
    assert np.array_equal(res.rgb, a)


# ------------------------------------------------------------ geometric

def test_lossless_rotate_and_flip(img):
    path, a = img
    src = engine.load(path)
    res = engine.execute(src, validate(raw_plan(step("rotate", degrees=-90), step("flip", direction="vertical"))))
    assert np.array_equal(res.rgb, np.flipud(np.rot90(a, -1)))


def test_crop_and_pad(img):
    path, a = img
    src = engine.load(path)
    res = engine.execute(src, validate(raw_plan(step("crop", x0=0.25, y0=0.5, x1=0.75, y1=1.0),
                                                step("pad", top=3, right=3, bottom=3, left=3, color="#000000"))))
    assert res.rgb.shape == (66, 86, 3)
    assert np.array_equal(res.rgb[3:-3, 3:-3], a[60:120, 40:120])
    assert (res.rgb[0] == 0).all()


def test_rotate_crop_mode_has_no_fill_corners(img):
    path, _ = img
    src = engine.load(path)
    res = engine.execute(src, validate(raw_plan(step("rotate", degrees=10, mode="crop", fill="#ff00ff"))))
    corners = res.rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert not ((corners[:, 0] > 240) & (corners[:, 1] < 15) & (corners[:, 2] > 240)).any()
    assert res.rgb.shape[0] < 120


def test_local_edit_then_resize_keeps_far_pixels_identical_to_resized_source(img):
    path, _ = img
    src = engine.load(path)
    res = engine.execute(src, validate(raw_plan(step("fill", region(box=(0, 0, 0.2, 0.2)), color="#ff0000"),
                                                step("resize", scale=0.5))))
    assert res.rgb.shape == (60, 80, 3)
    assert res.untouched_identical
    assert np.array_equal(res.rgb[40:, 40:], res.ref[40:, 40:])


# ------------------------------------------------------------ validation

def test_validator_rejects_and_reports():
    plan = validate(raw_plan(
        step("brightness", amount=5),
        step("nonexistent"),
        step("blur", clause=7, radius=2),
        step("resize"),
        clauses=("brighter", "other", "third"),
    ))
    assert [s.op for s in plan.steps] == ["brightness"]
    assert plan.steps[0].params["amount"] == 1.0
    text = "\n".join(plan.issues)
    assert "clamped" in text and "unknown operation" in text and "not part of the request" in text
    assert "clause 3" in text or "clause 2" in text


def test_validator_ignores_region_on_geometric():
    plan = validate(raw_plan(step("flip", region(), direction="horizontal")))
    assert plan.steps[0].region.is_full
    assert any("region ignored" in i for i in plan.issues)


def test_schema_and_catalog_stable():
    assert catalog_text() == catalog_text()
    assert set(PLAN_SCHEMA["properties"]["steps"]["items"]["properties"]["op"]["enum"]) == set(OPS)
    assert "Operations" in SYSTEM_PROMPT


# ------------------------------------------------------------ local planner

@pytest.mark.parametrize("prompt,ops", [
    ("rotate 90 degrees clockwise", ["rotate"]),
    ("Make it black and white", ["grayscale"]),
    ("flip horizontally and make it a bit brighter", ["flip", "brightness"]),
    ("increase contrast by 20%", ["contrast"]),
    ("crop to 16:9", ["crop"]),
    ("resize to 50%", ["resize"]),
    ("slightly warmer, sharpen", ["temperature", "sharpen"]),
])
def test_local_planner_handles_simple(prompt, ops):
    raw = plan_locally(prompt, (1600, 1200))
    assert raw is not None
    plan = validate(raw, "local")
    assert [s.op for s in plan.steps] == ops
    assert not plan.issues


@pytest.mark.parametrize("prompt", ["make the sky bluer", "remove the person on the left",
                                    "brighten the face", "rotate 90 and remove the car"])
def test_local_planner_defers_anything_partial(prompt):
    assert plan_locally(prompt, (100, 100)) is None


def test_local_planner_direction_words():
    p = validate(plan_locally("increase contrast", (10, 10)))
    assert p.steps[0].params["amount"] > 0
    p = validate(plan_locally("reduce saturation", (10, 10)))
    assert p.steps[0].params["amount"] < 0


# ------------------------------------------------------------ model planner (fake client)

class FakeClient:
    def __init__(self, payload, stop="end_turn"):
        self.calls = []
        self.payload, self.stop = payload, stop
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason=self.stop, stop_details=SimpleNamespace(category="cyber"),
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=900, output_tokens=300, cache_read_input_tokens=2000,
                                  cache_creation_input_tokens=0))


def test_model_planner_request_shape_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEDIT_CACHE", str(tmp_path / "cache"))
    payload = raw_plan(step("saturation", region(), amount=0.3), clauses=("bluer sky",))
    fc = FakeClient(payload)
    raw, usage = plan_with_model(b"jpegbytes", "facts", "make the sky bluer", client=fc)
    assert raw == payload and usage.output_tokens == 300
    kw = fc.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"]["format"]["schema"] is PLAN_SCHEMA
    assert kw["fallbacks"] == "default"
    raw2, usage2 = plan_with_model(b"jpegbytes", "facts", "make the sky bluer", client=fc)
    assert usage2.cached_plan and len(fc.calls) == 1


def test_model_planner_refusal(tmp_path, monkeypatch):
    from imgedit.planner_llm import PlannerError
    monkeypatch.setenv("IMGEDIT_CACHE", str(tmp_path / "cache"))
    with pytest.raises(PlannerError, match="declined"):
        plan_with_model(b"x", "f", "p", client=FakeClient({}, stop="refusal"))


# ------------------------------------------------------------ CLI / IO

def test_cli_apply_roundtrip_preserves_metadata(tmp_path):
    a = np.full((40, 50, 3), 100, np.uint8)
    src = tmp_path / "in.jpg"
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    Image.fromarray(a).save(src, quality=90, exif=exif, dpi=(300, 300))
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps(raw_plan(step("brightness", amount=0.2))))
    out = tmp_path / "out.jpg"
    assert main(["apply", str(src), str(plan), "-o", str(out), "--yes"]) == 0
    im = Image.open(out)
    assert im.getexif().get(0x010F) == "TestCam"
    assert round(im.info["dpi"][0]) == 300
    assert np.asarray(im).mean() > 110


def test_cli_refuses_to_overwrite_input(tmp_path, img):
    path, _ = img
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps(raw_plan(step("invert"))))
    assert main(["apply", path, str(plan), "-o", path, "--yes"]) == 1


def test_cli_plan_only_writes_nothing(tmp_path, img):
    path, _ = img
    assert main(["edit", path, "make it black and white", "--plan-only", "--offline"]) == 0
    assert not os.path.exists(path.replace(".png", ".edited.png"))


def test_alpha_preserved_png(tmp_path):
    a = np.zeros((20, 20, 4), np.uint8)
    a[..., :3] = 90
    a[5:15, 5:15, 3] = 255
    p = tmp_path / "a.png"
    Image.fromarray(a, "RGBA").save(p)
    src = engine.load(str(p))
    res = engine.execute(src, validate(raw_plan(step("brightness", amount=0.5))))
    out = tmp_path / "o.png"
    engine.save(src, res, str(out))
    b = np.asarray(Image.open(out))
    assert b.shape == (20, 20, 4) and np.array_equal(b[..., 3], a[..., 3])


def test_exif_orientation_applied(tmp_path):
    a = np.zeros((10, 20, 3), np.uint8)
    a[:, :10] = 255
    exif = Image.Exif()
    exif[0x0112] = 6
    p = tmp_path / "o.jpg"
    Image.fromarray(a).save(p, exif=exif, quality=95)
    src = engine.load(str(p))
    assert src.size == (10, 20)
    assert any("orientation" in n for n in src.notes)


def test_risk_report_flags_clipping_and_upscale(img):
    path, _ = img
    src = engine.load(path)
    plan = validate(raw_plan(step("exposure", stops=3), step("resize", scale=3.0)))
    res = engine.execute(src, plan)
    measured, possible = engine.analyse(src, plan, res, "PNG")
    joined = "\n".join(measured + possible)
    assert "upscaled 3.00x" in joined
    plan2 = validate(raw_plan(step("exposure", stops=3)))
    measured2, _ = engine.analyse(src, plan2, engine.execute(src, plan2), "PNG")
    assert any("clipped" in m for m in measured2)


def test_jpeg_quality_estimate(tmp_path):
    p = tmp_path / "q.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32, 3), np.uint8)).save(p, quality=75)
    assert abs(engine.estimate_jpeg_quality(Image.open(p)) - 75) <= 2
