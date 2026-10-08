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
