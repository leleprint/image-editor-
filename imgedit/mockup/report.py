"""Plan + limits + measured-risk report for mockups."""

from __future__ import annotations

from .engine import LIMITS
from .geometry import Cylinder
from .plan import MockupPlan


def _where(spec) -> str:
    pl = spec.placement
    if isinstance(spec.surface, Cylinder):
        return (f"cylinder, centre {pl.center_deg:+.0f}°, arc {pl.arc_deg:.0f}°, "
                f"axis {pl.v_range[0]:.0%}-{pl.v_range[1]:.0%}")
    kind = "fabric" if spec.surface.fabric else "flat face"
    u0, v0, u1, v1 = pl.uv_box
    box = "" if (u0, v0, u1, v1) == (0, 0, 1, 1) else f", region u {u0:.2f}-{u1:.2f} v {v0:.2f}-{v1:.2f}"
    return f"{kind}{box}{', corners snapped' if spec.snap else ''}"


def render(plan: MockupPlan, planner: str, usage: str, measured: list[str], possible: list[str]) -> str:
    out = [f"MOCKUP PLAN  [planner: {planner}{' | ' + usage if usage else ''}]"]
    if plan.understanding:
        out.append(f"Understood: {plan.understanding}")
    pr = plan.product
    if pr:
        out.append(f"Product: {pr.get('type', '?')}, {pr.get('color', '?')}, {pr.get('material', '?')}, {pr.get('finish', '?')}")
    out.append("")
    out.append("Requested:")
    uns: dict[int, list] = {}
    for u in plan.unsupported:
        uns.setdefault(int(u.get("clause", -1)), []).append(u)
    for ci, c in enumerate(plan.clauses):
        out.append(f"  {ci + 1}. \"{c}\"")
        for si, s in enumerate(plan.placements, 1):
            if s.clause != ci:
                continue
            size = f", {s.size['logo_width_mm']:.0f}mm wide" if s.size else ""
            out.append(f"     -> [{si}] {s.standard}{size}: {s.look.technique}, ink {s.ink}"
                       f"{' ' + s.ink_color if s.ink_color else ''} on {_where(s)}")
            out.append(f"            look: opacity {s.look.opacity:.2f}, texture {s.look.texture:.2f}, gloss {s.look.gloss:.2f}"
                       f"{', ' + str(len(s.occluders)) + ' occluder(s)' if s.occluders else ''}")
            if s.why:
                out.append(f"            why: {s.why}")
        for u in uns.get(ci, []):
            out.append(f"     x NOT DONE: {u.get('reason', '')}")
            if u.get("alternative"):
                out.append(f"            closest possible: {u['alternative']}")
    for u in uns.get(-1, []):
        out.append(f"  x NOT DONE: {u.get('reason', '')}")
    if not plan.placements:
        out.append("  (nothing to render - nothing will be written)")
    if plan.ambiguities:
        out += ["", "Assumptions made:"] + [f"  - {a.get('issue', '')} -> {a.get('assumption', '')}" for a in plan.ambiguities]
    if plan.preserved:
        out += ["", "Left unchanged:"] + [f"  - {p}" for p in plan.preserved]
    out += ["", "Limits and artifact risks:"]
    if measured:
        out += ["  measured on the full-resolution render:"] + [f"    - {m}" for m in measured]
    if plan.risks:
        out += ["  planner (photo-specific):"] + [f"    - {r}" for r in plan.risks]
    if possible:
        out += ["  possible:"] + [f"    - {p}" for p in possible]
    out += ["  system limits:"] + [f"    - {l}" for l in LIMITS]
    if plan.issues:
        out += ["", "Validator:"] + [f"  ! {i}" for i in plan.issues]
    if plan.needs_clarification and plan.question:
        out += ["", f"QUESTION: {plan.question}"]
    return "\n".join(out)
