"""Edit plan model, JSON schema for the planner, and strict validation.

Validation never guesses silently: out-of-range values are clamped *and
reported*, invalid steps are dropped *and reported*, and every requested
clause must map to at least one step or an explicit "unsupported" entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .ops import OPS, Op
from .pixels import parse_color
from .regions import SHAPES, ColorRange, Region


@dataclass
class Step:
    op: str
    params: dict[str, Any]
    region: Region
    clause: int
    why: str = ""

    def describe(self) -> str:
        op = OPS[self.op]
        shown = {k: v for k, v in self.params.items() if v != op.param(k).default or op.param(k).required}
        args = ", ".join(f"{k}={_fmt(v)}" for k, v in shown.items())
        where = "" if op.kind == "geometric" else f" on {self.region.describe()}"
        return f"{self.op}({args}){where}"


@dataclass
class Plan:
    understanding: str
    clauses: list[str]
    steps: list[Step]
    ambiguities: list[dict[str, str]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    question: str = ""
    issues: list[str] = field(default_factory=list)  # validator findings
    source: str = "llm"  # llm | local | file

    def to_json(self) -> str:
        d = asdict(self)
        for s in d["steps"]:
            s["region"] = _region_to_json(s["region"])
            s["params"] = [{"name": k, "value": v} for k, v in s["params"].items()]
        d.pop("issues")
        return json.dumps(d, indent=2)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, str):
        return repr(v)
    return str(v)


def _region_to_json(r: dict) -> Optional[dict]:
    r = dict(r)
    r.pop("notes", None)
    return r


# ----------------------------------------------------------- planner schema

_NUM = {"type": "number"}
_REGION = {
    "type": "object",
    "properties": {
        "shape": {"type": "string", "enum": list(SHAPES)},
        "box": {"anyOf": [{"type": "array", "items": _NUM}, {"type": "null"}]},
        "points": {"anyOf": [{"type": "array", "items": {"type": "array", "items": _NUM}}, {"type": "null"}]},
        "feather": _NUM,
        "invert": {"type": "boolean"},
        "color_range": {"anyOf": [{
            "type": "object",
            "properties": {k: _NUM for k in ("hue_center", "hue_tolerance", "min_saturation", "lum_min", "lum_max")},
            "required": ["hue_center", "hue_tolerance", "min_saturation", "lum_min", "lum_max"],
            "additionalProperties": False,
        }, {"type": "null"}]},
    },
    "required": ["shape", "box", "points", "feather", "invert", "color_range"],
    "additionalProperties": False,
}


def _obj(props: dict, ) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
PLAN_SCHEMA = _obj({
    "understanding": _STR,
    "clauses": {"type": "array", "items": _STR},
    "ambiguities": {"type": "array", "items": _obj({"issue": _STR, "assumption": _STR})},
    "needs_clarification": {"type": "boolean"},
    "question": _STR,
    "steps": {"type": "array", "items": _obj({
        "clause": {"type": "integer"},
        "op": {"type": "string", "enum": sorted(OPS)},
        "params": {"type": "array", "items": _obj({
            "name": _STR,
            "value": {"anyOf": [_NUM, _STR, {"type": "boolean"}]},
        })},
        "region": _REGION,
        "why": _STR,
    })},
    "unsupported": {"type": "array", "items": _obj({"clause": {"type": "integer"}, "reason": _STR, "alternative": _STR})},
    "preserved": {"type": "array", "items": _STR},
    "risks": {"type": "array", "items": _STR},
})


# --------------------------------------------------------------- validation

def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _parse_region(raw: Optional[dict], op: Op, issues: list[str], tag: str) -> Region:
    if not raw or op.kind == "geometric":
        if raw and raw.get("shape", "full") != "full" and op.kind == "geometric":
            issues.append(f"{tag}: region ignored - geometric ops always apply to the whole canvas")
        return Region()
    shape = raw.get("shape", "full")
    if shape not in SHAPES:
        raise ValueError(f"unknown region shape {shape!r}")
    region = Region(shape=shape, invert=bool(raw.get("invert", False)))
    if shape in ("rect", "ellipse"):
        box = raw.get("box")
        if not box or len(box) != 4:
            raise ValueError(f"{shape} region needs box [x0,y0,x1,y1]")
        x0, y0, x1, y1 = (_clamp01(v) for v in box)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if (x1 - x0) < 1e-3 or (y1 - y0) < 1e-3:
            raise ValueError("region box is empty")
        if [x0, y0, x1, y1] != [float(v) for v in box]:
            issues.append(f"{tag}: region box clamped/reordered to [{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}]")
        region.box = [x0, y0, x1, y1]
    elif shape == "polygon":
        pts = raw.get("points") or []
        if len(pts) < 3 or any(len(p) != 2 for p in pts):
            raise ValueError("polygon region needs >=3 [x,y] points")
        region.points = [[_clamp01(x), _clamp01(y)] for x, y in pts]
    feather = float(raw.get("feather") or 0.0)
    if not 0 <= feather <= 0.25:
        issues.append(f"{tag}: feather {feather} clamped to 0..0.25")
        feather = min(0.25, max(0.0, feather))
    region.feather = feather
    cr = raw.get("color_range")
    if cr:
        region.color_range = ColorRange(
            hue_center=float(cr["hue_center"]) % 360,
            hue_tolerance=min(180.0, max(1.0, float(cr.get("hue_tolerance", 25)))),
            min_saturation=_clamp01(cr.get("min_saturation", 0.15)),
            lum_min=_clamp01(cr.get("lum_min", 0.0)),
            lum_max=_clamp01(cr.get("lum_max", 1.0)),
        )
    if region.shape == "full" and region.invert and region.color_range is None:
        raise ValueError("inverted full region selects nothing")
    return region


def _coerce(op: Op, name: str, value: Any, issues: list[str], tag: str) -> Any:
    p = op.param(name)
    if p.type in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{name} must be a number")
        v = float(value)
        if v != v:  # NaN
            raise ValueError(f"{name} is NaN")
        lo, hi = p.min, p.max
        if v < lo or v > hi:
            issues.append(f"{tag}: {name}={v:g} outside {lo:g}..{hi:g}, clamped")
            v = min(hi, max(lo, v))
        return int(round(v)) if p.type == "int" else v
    if p.type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes")
        return bool(value)
    if p.type == "enum":
        if value not in p.choices:
            raise ValueError(f"{name} must be one of {', '.join(p.choices)}")
        return value
    if p.type == "color":
        if str(value).lower() != "transparent":
            parse_color(str(value))
        return str(value)
    return str(value)


def validate(raw: dict, source: str = "llm") -> Plan:
    issues: list[str] = []
    clauses = [str(c) for c in raw.get("clauses", [])]
    steps: list[Step] = []
    covered: set[int] = set()
    unsupported = list(raw.get("unsupported", []))

    for i, s in enumerate(raw.get("steps", [])):
        tag = f"step {i + 1} ({s.get('op')})"
        try:
            op = OPS.get(s.get("op"))
            if op is None:
                raise ValueError("unknown operation")
            clause = int(s.get("clause", -1))
            if not 0 <= clause < len(clauses):
                raise ValueError(f"references clause {clause}, which is not part of the request")
            given = s.get("params", [])
            if isinstance(given, dict):
                given = [{"name": k, "value": v} for k, v in given.items()]
            params: dict[str, Any] = {}
            for kv in given:
                name = kv.get("name")
                if name not in {p.name for p in op.params}:
                    issues.append(f"{tag}: unknown parameter {name!r} ignored")
                    continue
                params[name] = _coerce(op, name, kv.get("value"), issues, tag)
            for p in op.params:
                if p.name not in params:
                    if p.required:
                        raise ValueError(f"missing required parameter {p.name}")
                    params[p.name] = p.default
            if op.name == "crop":
                for a, b in (("x0", "x1"), ("y0", "y1")):
                    if params[b] - params[a] < 0.01:
                        raise ValueError("crop box is empty")
            if op.name == "resize" and not (params["scale"] or params["width"] or params["height"]):
                raise ValueError("resize needs scale, width or height")
            if op.name == "text" and not params["text"].strip():
                raise ValueError("text is empty")
            region = _parse_region(s.get("region"), op, issues, tag)
            steps.append(Step(op=op.name, params=params, region=region, clause=clause, why=str(s.get("why", ""))))
            covered.add(clause)
        except (ValueError, KeyError, TypeError) as e:
            issues.append(f"{tag} REJECTED, not executed: {e}")
            unsupported.append({"clause": int(s.get("clause", -1)) if str(s.get("clause", "")).lstrip("-").isdigit() else -1,
                                "reason": f"planned step was invalid ({e})", "alternative": ""})

    for u in unsupported:
        covered.add(int(u.get("clause", -1)))
    for ci, c in enumerate(clauses):
        if ci not in covered:
            issues.append(f"clause {ci + 1} ({c!r}) has no step and was not marked unsupported")

    geo_seen = False
    for i, st in enumerate(steps):
        if OPS[st.op].kind == "geometric":
            geo_seen = True
        elif geo_seen and not st.region.is_full:
            issues.append(f"step {i + 1}: region coordinates refer to the canvas after the preceding geometric step(s)")

    return Plan(
        understanding=str(raw.get("understanding", "")),
        clauses=clauses,
        steps=steps,
        ambiguities=list(raw.get("ambiguities", [])),
        unsupported=unsupported,
        preserved=[str(p) for p in raw.get("preserved", [])],
        risks=[str(r) for r in raw.get("risks", [])],
        needs_clarification=bool(raw.get("needs_clarification", False)),
        question=str(raw.get("question", "")),
        issues=issues,
        source=source,
    )


def load_plan(path: str) -> Plan:
    with open(path, encoding="utf-8") as f:
        return validate(json.load(f), source="file")
