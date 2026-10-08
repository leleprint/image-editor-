"""Human-readable plan + limits + risk report, shown before anything is written."""

from __future__ import annotations

from .ops import OPS
from .plan import Plan

LIMITS = (
    "Edits are deterministic adjustments only: nothing is generated, removed, moved or redrawn.",
    "Regions are geometric shapes and colour ranges, not object segmentation; boundaries are approximate.",
    "Pixels outside the planned regions are locked byte-for-byte to the source (verified after the run).",
)


def render(plan: Plan, planner: str, usage_line: str, measured: list[str], possible: list[str]) -> str:
    out: list[str] = []
    out.append(f"PLAN  [planner: {planner}{' | ' + usage_line if usage_line else ''}]")
    if plan.understanding:
        out.append(f"Understood: {plan.understanding}")
    out.append("")
    out.append("Requested changes:")
    unsupported = {}
    for u in plan.unsupported:
        unsupported.setdefault(int(u.get("clause", -1)), []).append(u)
    for ci, clause in enumerate(plan.clauses):
        out.append(f"  {ci + 1}. \"{clause}\"")
        for si, s in enumerate(plan.steps, 1):
            if s.clause == ci:
                out.append(f"     -> [{si}] {s.describe()}")
                if s.why:
                    out.append(f"            why: {s.why}")
        for u in unsupported.get(ci, []):
            out.append(f"     x NOT DONE: {u.get('reason', '')}")
            if u.get("alternative"):
                out.append(f"            closest possible: {u['alternative']}")
    for u in unsupported.get(-1, []):
        out.append(f"  x NOT DONE: {u.get('reason', '')}")
    if not plan.steps:
        out.append("  (no executable steps - nothing will be written)")

    if plan.ambiguities:
        out.append("")
        out.append("Assumptions made:")
        for a in plan.ambiguities:
            out.append(f"  - {a.get('issue', '')} -> {a.get('assumption', '')}")
    if plan.preserved:
        out.append("")
        out.append("Left unchanged:")
        for p in plan.preserved:
            out.append(f"  - {p}")

    out.append("")
    out.append("Limits and artifact risks:")
    if measured:
        out.append("  measured on a full-resolution dry run:")
        out.extend(f"    - {m}" for m in measured)
    if plan.risks:
        out.append("  planner (image-specific):")
        out.extend(f"    - {r}" for r in plan.risks)
    if possible:
        out.append("  possible:")
        out.extend(f"    - {p}" for p in possible)
    out.append("  system limits:")
    out.extend(f"    - {l}" for l in LIMITS)
    if plan.issues:
        out.append("")
        out.append("Validator:")
        out.extend(f"  ! {i}" for i in plan.issues)
    if plan.needs_clarification and plan.question:
        out.append("")
        out.append(f"QUESTION: {plan.question}")
    return "\n".join(out)


def catalog_help() -> str:
    lines = []
    for op in OPS.values():
        params = ", ".join(
            f"{p.name}" + (f"[{p.min:g}..{p.max:g}]" if p.type in ("float", "int") else
                           f"[{'|'.join(p.choices)}]" if p.choices else f"<{p.type}>")
            for p in op.params)
        lines.append(f"{op.name:14} {op.kind:9} {op.summary}" + (f"\n{'':25}{params}" if params else ""))
    return "\n".join(lines)
