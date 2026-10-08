"""Mockup plan: schema for the planner, strict validation into executable specs.

Coordinates in the JSON are normalized to the product photo (x by width,
y by height; cylinder radii by width, ellipse half-heights by height).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from ..pixels import parse_color
from .geometry import Cylinder, Flat, Placement
from .placements import PLACEMENTS
from .render import Look

TECHNIQUES = ("print", "engrave", "emboss", "deboss")
INKS = ("original", "knockout_white", "single_color")
KINDS = ("flat", "fabric", "cylinder")
_N = {"type": "number"}
_PT = {"type": "array", "items": _N}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_S = {"type": "string"}
MOCKUP_SCHEMA = _obj({
    "understanding": _S,
    "clauses": {"type": "array", "items": _S},
    "ambiguities": {"type": "array", "items": _obj({"issue": _S, "assumption": _S})},
    "needs_clarification": {"type": "boolean"},
    "question": _S,
    "product": _obj({"type": _S, "color": _S,
                     "material": {"type": "string", "enum": ["cotton", "polyester", "knit", "canvas", "ceramic", "metal",
                                                             "glass", "wood", "paper", "cardboard", "plastic", "leather", "other"]},
                     "finish": {"type": "string", "enum": ["matte", "satin", "gloss"]}}),
    "placements": {"type": "array", "items": _obj({
        "clause": {"type": "integer"},
        "standard": _S,
        "why": _S,
        "surface": _obj({
            "kind": {"type": "string", "enum": list(KINDS)},
            "quad": _nullable({"type": "array", "items": _PT}),
            "snap": {"type": "boolean"},
            "cylinder": _nullable(_obj({"top": _PT, "bottom": _PT, "r_top": _N, "r_bottom": _N, "b_top": _N, "b_bottom": _N})),
        }),
        "region": _obj({"uv_box": _PT, "center_deg": _N, "arc_deg": _N, "v_range": _PT,
                        "anchor": {"type": "string", "enum": ["center", "top"]}}),
        "size": _nullable(_obj({"logo_width_mm": _N, "ref_p0": _PT, "ref_p1": _PT, "ref_length_mm": _N, "ref_what": _S})),
        "look": _obj({"technique": {"type": "string", "enum": list(TECHNIQUES)}, "opacity": _N, "gloss": _N, "texture": _N,
                      "ink": {"type": "string", "enum": list(INKS)}, "ink_color": _S}),
        "occluders": {"type": "array", "items": {"type": "array", "items": _PT}},
    })},
    "unsupported": {"type": "array", "items": _obj({"clause": {"type": "integer"}, "reason": _S, "alternative": _S})},
    "preserved": {"type": "array", "items": _S},
    "risks": {"type": "array", "items": _S},
})


@dataclass
class PlacementSpec:
    clause: int
    standard: str
    why: str
    surface: Any  # Flat | Cylinder (px)
    placement: Placement
    look: Look
    ink: str
    ink_color: str
    snap: bool
    size: Optional[dict]  # px-converted reference
    occluders: list[np.ndarray] = field(default_factory=list)


@dataclass
class MockupPlan:
    understanding: str
    clauses: list[str]
    product: dict
    placements: list[PlacementSpec]
    ambiguities: list[dict] = field(default_factory=list)
    unsupported: list[dict] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    question: str = ""
    issues: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.raw, indent=2)


def _c01(v, lo=0.0, hi=1.0) -> float:
    return float(min(hi, max(lo, float(v))))


def _order_quad(q: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL regardless of the order given."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise in image coords starting from ~ -180deg (left)
    s = q.sum(1)
    start = int(np.argmin(s))  # TL has smallest x+y
    return np.roll(q, -start, axis=0)


def validate_mockup(raw: dict, size: tuple[int, int]) -> MockupPlan:
    W, H = size
    issues: list[str] = []
    clauses = [str(c) for c in raw.get("clauses", [])]
    unsupported = list(raw.get("unsupported", []))
    specs: list[PlacementSpec] = []
    covered = {int(u.get("clause", -1)) for u in unsupported}

    for i, p in enumerate(raw.get("placements", [])):
        tag = f"placement {i + 1}"
        try:
            clause = int(p.get("clause", -1))
            if not 0 <= clause < len(clauses):
                raise ValueError(f"references clause {clause}, not part of the request")
            sf = p["surface"]
            kind = sf.get("kind")
            if kind not in KINDS:
                raise ValueError(f"unknown surface kind {kind!r}")
            rg = p.get("region") or {}
            uv = rg.get("uv_box") or [0, 0, 1, 1]
            if len(uv) != 4:
                raise ValueError("uv_box needs 4 numbers")
            uv = [_c01(v, -0.5, 1.5) for v in uv]
            if uv[2] - uv[0] < 0.01 or uv[3] - uv[1] < 0.01:
                raise ValueError("uv_box is empty")
            vr = rg.get("v_range") or [0.2, 0.8]
            vr = [_c01(vr[0]), _c01(vr[1])]
            if vr[1] - vr[0] < 0.02:
                raise ValueError("v_range is empty")
            arc = float(rg.get("arc_deg", 60))
            if not 2 <= arc <= 300:
                issues.append(f"{tag}: arc_deg {arc:g} clamped to 2..300")
                arc = min(300.0, max(2.0, arc))
            pl = Placement(uv_box=tuple(uv), center_deg=_c01(rg.get("center_deg", 0), -180, 180), arc_deg=arc,
                           v_range=tuple(vr), anchor=rg.get("anchor", "center") if rg.get("anchor") in ("center", "top") else "center")
            if kind in ("flat", "fabric"):
                q = np.array(sf.get("quad") or [], float)
                if q.shape != (4, 2):
                    raise ValueError(f"{kind} surface needs quad of 4 [x,y] points")
                q = np.clip(q, -0.25, 1.25) * [W, H]
                q = _order_quad(q)
                if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
                    raise ValueError("quad is not convex")
                if cv2.contourArea(q.astype(np.float32).reshape(-1, 1, 2)) < 64:
                    raise ValueError("quad is too small")
                surface = Flat(q, fabric=(kind == "fabric"))
            else:
                c = sf.get("cylinder")
                if not c:
                    raise ValueError("cylinder surface needs cylinder geometry")
                top = np.array(c["top"], float) * [W, H]
                bot = np.array(c["bottom"], float) * [W, H]
                if np.linalg.norm(bot - top) < 8:
                    raise ValueError("cylinder axis too short")
                rt, rb = float(c["r_top"]) * W, float(c["r_bottom"]) * W
                if min(rt, rb) < 4:
                    raise ValueError("cylinder radius too small")
                bt, bb = float(c["b_top"]) * H, float(c["b_bottom"]) * H
                for name, b, r in (("b_top", bt, rt), ("b_bottom", bb, rb)):
                    if abs(b) > 0.95 * r:
                        issues.append(f"{tag}: {name} larger than radius (view from nearly straight above?) - clamped")
                bt = float(np.clip(bt, -0.95 * rt, 0.95 * rt))
                bb = float(np.clip(bb, -0.95 * rb, 0.95 * rb))
                surface = Cylinder(top, bot, rt, rb, bt, bb)
                if abs(pl.center_deg) + arc / 2 > 80:
                    issues.append(f"{tag}: logo reaches {abs(pl.center_deg) + arc / 2:.0f}° around the cylinder - "
                                  "edges will be strongly foreshortened or hidden")
            lk = p.get("look") or {}
            tech = lk.get("technique", "print")
            if tech not in TECHNIQUES:
                raise ValueError(f"unknown technique {tech!r}")
            look = Look(technique=tech, opacity=_c01(lk.get("opacity", 0.97), 0.3, 1.0), gloss=_c01(lk.get("gloss", 0)),
                        texture=_c01(lk.get("texture", 0)))
            ink = lk.get("ink", "original")
            if ink not in INKS:
                raise ValueError(f"unknown ink mode {ink!r}")
            ink_color = str(lk.get("ink_color") or "")
            if ink == "single_color" or (tech == "engrave" and ink_color):
                parse_color(ink_color)
            if tech == "engrave" and ink_color:
                look.engrave_color = ink_color
            sz = p.get("size")
            size_px = None
            if sz and float(sz.get("ref_length_mm", 0)) > 0 and float(sz.get("logo_width_mm", 0)) > 0:
                p0 = np.array(sz["ref_p0"], float) * [W, H]
                p1 = np.array(sz["ref_p1"], float) * [W, H]
                d = float(np.linalg.norm(p1 - p0))
                if d < 5:
                    issues.append(f"{tag}: size reference too short - physical size not enforced")
                else:
                    size_px = {"px_per_mm": d / float(sz["ref_length_mm"]), "logo_width_mm": float(sz["logo_width_mm"]),
                               "ref_what": str(sz.get("ref_what", ""))}
            std = str(p.get("standard", "custom"))
            if std != "custom" and std not in PLACEMENTS:
                issues.append(f"{tag}: unknown standard {std!r}; treated as custom")
                std = "custom"
            if size_px and std in PLACEMENTS and PLACEMENTS[std].get("width"):
                lo, hi = PLACEMENTS[std]["width"]
                wmm = size_px["logo_width_mm"]
                if not lo * 0.9 <= wmm <= hi * 1.1:
                    issues.append(f"{tag}: requested {wmm:.0f}mm wide is outside the usual {lo:.0f}-{hi:.0f}mm for {std}")
            occ = []
            for poly in p.get("occluders", []) or []:
                arr = np.array(poly, float)
                if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                    occ.append(np.clip(arr, 0, 1) * [W, H])
            specs.append(PlacementSpec(clause, std, str(p.get("why", "")), surface, pl, look, ink, ink_color,
                                       bool(sf.get("snap", False)), size_px, occ))
            covered.add(clause)
        except (KeyError, TypeError, ValueError) as e:
            issues.append(f"{tag} REJECTED, not rendered: {e}")
            unsupported.append({"clause": int(p.get("clause", -1)) if isinstance(p.get("clause"), int) else -1,
                                "reason": f"planned placement was invalid ({e})", "alternative": ""})
    for ci, c in enumerate(clauses):
        if ci not in covered:
            issues.append(f"clause {ci + 1} ({c!r}) has no placement and was not marked unsupported")
    return MockupPlan(
        understanding=str(raw.get("understanding", "")), clauses=clauses, product=dict(raw.get("product") or {}),
        placements=specs, ambiguities=list(raw.get("ambiguities", [])), unsupported=unsupported,
        preserved=[str(x) for x in raw.get("preserved", [])], risks=[str(x) for x in raw.get("risks", [])],
        needs_clarification=bool(raw.get("needs_clarification", False)), question=str(raw.get("question", "")),
        issues=issues, raw=raw,
    )
