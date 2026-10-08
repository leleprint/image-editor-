"""Command line: plan -> validate -> dry run -> report -> confirm -> write."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from . import engine
from .plan import Plan, load_plan, validate
from .planner_local import plan_locally
from .report import catalog_help, render

EXIT_OK, EXIT_ERROR, EXIT_CLARIFY, EXIT_NOTHING = 0, 1, 2, 3


def _default_output(inp: str) -> str:
    stem, ext = os.path.splitext(inp)
    return f"{stem}.edited{ext}"


def _plan(src: engine.Source, prompt: str, args) -> tuple[Plan, str, str]:
    if not args.force_model:
        raw = plan_locally(prompt, src.size)
        if raw is not None:
            return validate(raw, source="local"), "local rules", "tokens: 0"
    if args.offline:
        raise SystemExit("error: this request needs the model planner (remove --offline)")
    from .planner_llm import PlannerError, plan_with_model

    preview, _ = engine.preview_jpeg(src, args.preview_size)
    try:
        raw, usage = plan_with_model(preview, engine.image_facts(src), prompt, model=args.model,
                                     effort=args.effort, use_cache=not args.no_cache)
    except PlannerError as e:
        raise SystemExit(f"error: {e}")
    return validate(raw, source="llm"), args.model, usage.line()


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _run(src: engine.Source, plan: Plan, planner: str, usage: str, args, inp: str) -> int:
    out = args.output or (inp if args.in_place else _default_output(inp))
    out_fmt = engine.format_for(out, src.format)
    res = engine.execute(src, plan, dither=args.dither) if plan.steps else None
    measured, possible = engine.analyse(src, plan, res, out_fmt) if res else ([], [])

    if args.json:
        print(json.dumps({"plan": json.loads(plan.to_json()), "issues": plan.issues, "measured": measured,
                          "possible": possible, "planner": planner, "usage": usage}, indent=2))
    else:
        print(render(plan, planner, usage, measured, possible))
        print()

    if args.save_plan:
        with open(args.save_plan, "w", encoding="utf-8") as f:
            f.write(plan.to_json())
        print(f"plan saved to {args.save_plan} (re-apply with: imgedit apply {inp} {args.save_plan})")

    if res is None:
        return EXIT_NOTHING
    if not res.untouched_identical:
        print("error: scope verification failed; nothing written", file=sys.stderr)
        return EXIT_ERROR
    if args.plan_only:
        return EXIT_OK
    if os.path.abspath(out) == os.path.abspath(inp) and not args.in_place:
        print("error: output would overwrite the input; pass --in-place to allow", file=sys.stderr)
        return EXIT_ERROR
    if not args.yes and not _confirm(f"Write {out}?"):
        print("not written")
        return EXIT_NOTHING
    info = engine.save(src, res, out, quality=args.quality)
    extra = ", ".join(f"{k}={v}" for k, v in info.items() if k != "format")
    print(f"wrote {out} ({info['format']}{', ' + extra if extra else ''})")
    return EXIT_OK


def cmd_edit(args) -> int:
    src = engine.load(args.input)
    prompt = args.prompt
    for _ in range(3):
        plan, planner, usage = _plan(src, prompt, args)
        if not (plan.needs_clarification and plan.question):
            break
        if args.yes or not sys.stdin.isatty():
            print(render(plan, planner, usage, [], []))
            print("\nThe request is ambiguous. Re-run with a more specific prompt "
                  "(or answer interactively without --yes).", file=sys.stderr)
            return EXIT_CLARIFY
        print(render(plan, planner, usage, [], []))
        answer = input("\nYour answer (empty = accept the assumptions above): ").strip()
        if not answer:
            plan.needs_clarification = False
            break
        prompt = f"{args.prompt}\nClarification from user: {answer}"
    return _run(src, plan, planner, usage, args, args.input)


def cmd_apply(args) -> int:
    src = engine.load(args.input)
    plan = load_plan(args.plan)
    return _run(src, plan, "saved plan", "tokens: 0", args, args.input)


def _mockup_run(args, src, logo, plan, planner: str, usage: str) -> int:
    from .mockup import engine as mk
    from .mockup.report import render as mrender

    res = mk.run(src.u8, logo, plan, refine=not args.no_refine) if plan.placements else None
    measured, possible = mk.analyse(src.u8, logo, plan, res) if res else ([], [])
    if args.json:
        print(json.dumps({"plan": plan.raw, "issues": plan.issues, "measured": measured, "possible": possible,
                          "planner": planner, "usage": usage}, indent=2))
    else:
        print(mrender(plan, planner, usage, measured, possible))
        print()
    if args.save_plan:
        with open(args.save_plan, "w", encoding="utf-8") as f:
            f.write(plan.to_json())
        print(f"plan saved to {args.save_plan} (re-render with: imgedit mockup-apply {args.product} {args.logo} {args.save_plan})")
    if res is None:
        return EXIT_NOTHING
    if not res.identical_outside:
        print("error: scope verification failed; nothing written", file=sys.stderr)
        return EXIT_ERROR
    out = args.output or _default_output(args.product).replace(".edited", ".mockup")
    if os.path.abspath(out) == os.path.abspath(args.product):
        print("error: output would overwrite the product photo", file=sys.stderr)
        return EXIT_ERROR
    if args.plan_only:
        if args.guide:
            g = os.path.splitext(out)[0] + ".guide.png"
            mk.guide_image(src.u8, plan, res).save(g)
            print(f"wrote placement guide {g}")
        return EXIT_OK
    if not args.yes and not _confirm(f"Write {out}?"):
        print("not written")
        return EXIT_NOTHING
    result = engine.Result(res.rgb, None, src.u8, res.footprint, [], 0, res.identical_outside, False)
    info = engine.save(src, result, out, quality=args.quality)
    print(f"wrote {out} ({info['format']})")
    if args.guide:
        g = os.path.splitext(out)[0] + ".guide.png"
        mk.guide_image(src.u8, plan, res).save(g)
        print(f"wrote placement guide {g}")
    return EXIT_OK


def cmd_mockup(args) -> int:
    from .mockup.logo import describe, load_logo, preview_png
    from .mockup.plan import validate_mockup
    from .mockup.planner import plan_mockup
    from .planner_llm import PlannerError

    src = engine.load(args.product)
    logo = load_logo(args.logo)
    preview, _ = engine.preview_jpeg(src, args.preview_size)
    facts = f"Product photo: {engine.image_facts(src)}\nLogo: {describe(logo)}"
    try:
        raw, usage = plan_mockup(preview, preview_png(logo), facts, args.request, model=args.model,
                                 effort=args.effort, use_cache=not args.no_cache)
    except PlannerError as e:
        raise SystemExit(f"error: {e}")
    plan = validate_mockup(raw, src.size)
    if plan.needs_clarification and plan.question and (args.yes or not sys.stdin.isatty()):
        from .mockup.report import render as mrender
        print(mrender(plan, args.model, usage.line(), [], []))
        print("\nThe request is ambiguous. Re-run with a more specific request.", file=sys.stderr)
        return EXIT_CLARIFY
    return _mockup_run(args, src, logo, plan, args.model, usage.line())


def cmd_mockup_apply(args) -> int:
    from .mockup.logo import load_logo
    from .mockup.plan import validate_mockup

    src = engine.load(args.product)
    logo = load_logo(args.logo)
    with open(args.plan, encoding="utf-8") as f:
        plan = validate_mockup(json.load(f), src.size)
    return _mockup_run(args, src, logo, plan, "saved plan", "tokens: 0")


def cmd_ops(_args) -> int:
    print(catalog_help())
    return EXIT_OK


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", help="output path (default: <name>.edited.<ext>; extension picks format)")
    p.add_argument("-y", "--yes", action="store_true", help="write without asking for confirmation")
    p.add_argument("--plan-only", action="store_true", help="show plan and risk report, write nothing")
    p.add_argument("--in-place", action="store_true", help="allow overwriting the input file")
    p.add_argument("--dither", action="store_true", help="dither edited pixels to suppress banding")
    p.add_argument("--quality", type=int, help="JPEG/WebP quality (default: max(92, source quality); WebP lossless)")
    p.add_argument("--save-plan", metavar="PATH", help="write the validated plan as JSON")
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="imgedit", description="Plan-first, scope-locked image editor.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("edit", help="edit an image from a natural-language request")
    e.add_argument("input")
    e.add_argument("prompt")
    _common(e)
    e.add_argument("--model", default=os.environ.get("IMGEDIT_MODEL", "claude-opus-5-5"))
    e.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    e.add_argument("--preview-size", type=int, default=768, help="long side of the preview sent to the model")
    e.add_argument("--offline", action="store_true", help="never call the model (local rules only)")
    e.add_argument("--force-model", action="store_true", help="skip the local fast path")
    e.add_argument("--no-cache", action="store_true", help="ignore the on-disk plan cache")
    e.set_defaults(fn=cmd_edit)

    a = sub.add_parser("apply", help="apply a saved plan (no model call)")
    a.add_argument("input")
    a.add_argument("plan")
    _common(a)
    a.set_defaults(fn=cmd_apply)

    def _mock_common(p):
        p.add_argument("-o", "--output", help="output path (default: <product>.mockup.<ext>)")
        p.add_argument("-y", "--yes", action="store_true", help="write without asking for confirmation")
        p.add_argument("--plan-only", action="store_true", help="show plan and risk report, write nothing")
        p.add_argument("--guide", action="store_true", help="also write <out>.guide.png with the surface/placement grid")
        p.add_argument("--no-refine", action="store_true", help="don't snap geometry to image edges")
        p.add_argument("--quality", type=int, help="JPEG/WebP quality")
        p.add_argument("--save-plan", metavar="PATH", help="write the plan JSON for re-use / manual tweaks")
        p.add_argument("--json", action="store_true", help="print the report as JSON")

    m = sub.add_parser("mockup", help="place a logo on a product photo realistically")
    m.add_argument("product")
    m.add_argument("logo", help="SVG (preferred) or PNG with transparency")
    m.add_argument("request", help='e.g. "left chest print" or "centred on the mug, 70mm wide"')
    _mock_common(m)
    m.add_argument("--model", default=os.environ.get("IMGEDIT_MODEL", "claude-opus-5-5"))
    m.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    m.add_argument("--preview-size", type=int, default=1024, help="long side of the product preview sent to the model")
    m.add_argument("--no-cache", action="store_true", help="ignore the on-disk plan cache")
    m.set_defaults(fn=cmd_mockup)

    ma = sub.add_parser("mockup-apply", help="render a saved mockup plan (no model call)")
    ma.add_argument("product")
    ma.add_argument("logo")
    ma.add_argument("plan")
    _mock_common(ma)
    ma.set_defaults(fn=cmd_mockup_apply)

    o = sub.add_parser("ops", help="list available operations")
    o.set_defaults(fn=cmd_ops)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
