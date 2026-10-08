"""Model planner: one structured call turns a request into a scoped plan.

Token budget levers, in order of impact:
  1. Local fast path (planner_local) - zero tokens for simple requests.
  2. Plan cache keyed on (image hash, prompt, model) - zero tokens on reruns.
  3. Downscaled JPEG preview (default 768px long side, ~500-800 image tokens)
     plus a one-line numeric summary of the full-resolution image.
  4. Frozen system prompt + op catalog marked for prompt caching.
  5. Compact JSON schema output (no prose), effort configurable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from .ops import catalog_text
from .plan import PLAN_SCHEMA

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = f"""You are the planning stage of a precise, non-generative image editor. You never edit pixels; you return a JSON plan that a deterministic engine executes exactly. The user sees your plan before anything is written.

Goals, in priority order:
1. Do exactly what was asked - nothing more. Never add "improvements" the user did not request (no auto-enhance, no extra sharpening, no crop, no colour fixes). If the request says "brighten the sky", nothing outside the sky may change.
2. Understand intent precisely. Split the request into atomic clauses (one requested change each, in the user's words). Every step cites the clause it serves; every clause gets steps or an "unsupported" entry.
3. Be honest about limits. The engine has only the operations below. It cannot generate, remove, move or redraw objects, replace backgrounds, retouch faces, change expressions, restore real detail, or segment objects semantically. Requests needing that go in "unsupported" with a concrete reason and the closest achievable alternative (alternative = "" if none). Do not approximate an unsupported request with a misleading operation (e.g. do not "remove" a person with a blur or fill unless the user asked to hide/redact it).
4. Scope every local edit tightly. Regions: rect/ellipse box=[x0,y0,x1,y1] or polygon points=[[x,y],...], all normalized 0..1 to the canvas at that step (origin top-left). Estimate boxes from the preview; prefer a polygon for irregular subjects. Use feather (0..0.25, fraction of short side; mask is 50% at the boundary) 0.01-0.05 for natural tonal edits, 0 for redaction fill/pixelate. Add color_range (hue_center deg, hue_tolerance deg, min_saturation, lum_min, lum_max) to restrict to a colour such as sky blue (hue~210) or foliage (hue~100); use lum bounds to target bright or dark areas. invert=true selects everything outside the shape. Use shape "full" with box/points null for whole-image edits. Geometric ops ignore regions.
5. Pick conservative, faithful magnitudes. Mild words ("slightly", "a bit") -> small values (|amount| 0.1-0.2); unqualified -> moderate (0.2-0.35); strong words -> 0.4-0.7. Prefer the op that avoids artifacts: brightness/shadows/highlights over exposure when clipping is a concern; vibrance over saturation for natural colour; sharpen with threshold>0 on noisy images.
6. Order steps sensibly: local/tonal edits first, then crop/rotate/resize, then text overlays. Coordinates of steps after a geometric op refer to the transformed canvas.
7. If the request is genuinely ambiguous in a way that changes the result materially and no reasonable default exists, set needs_clarification=true, ask ONE short question, and still return the best-guess plan. Otherwise record each assumption in "ambiguities" and proceed.
8. "preserved": short list of what will stay untouched (e.g. "subject's skin tones", "image dimensions"). "risks": artifacts or failure modes specific to THIS image and plan (e.g. "sky has smooth gradient - banding possible", "box for the sign is approximate; edges of nearby wall may lighten"). No generic filler. Keep all text terse.

Operations (name(param*=required:range or param=default:range)):
{catalog_text()}

Params are given as [{{"name":..., "value":...}}]; omit params left at default. Colours are "#rrggbb" (or "transparent" for rotate/pad fill)."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cached_plan: bool = False

    def line(self) -> str:
        if self.cached_plan:
            return "tokens: 0 (plan cache hit)"
        return (f"tokens: in {self.input_tokens} (+cache read {self.cache_read}, cache write {self.cache_write}), "
                f"out {self.output_tokens}")


class PlannerError(RuntimeError):
    pass


def _cache_dir() -> str:
    d = os.environ.get("IMGEDIT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "imgedit"))
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(image_bytes: bytes, prompt: str, model: str, effort: str, facts: str) -> str:
    h = hashlib.sha256()
    for part in (image_bytes, prompt.strip().encode(), model.encode(), effort.encode(), facts.encode(),
                 hashlib.sha256(SYSTEM_PROMPT.encode()).digest()):
        h.update(part)
        h.update(b"\0")
    return h.hexdigest()[:32]


def plan_with_model(
    preview: bytes,
    facts: str,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    use_cache: bool = True,
    client=None,
) -> tuple[dict, Usage]:
    key = _cache_key(preview, prompt, model, effort, facts)
    path = os.path.join(_cache_dir(), f"{key}.json") if use_cache else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), Usage(cached_plan=True)

    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    import anthropic  # noqa: F811 - typed errors

    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(preview).decode()}},
                    {"type": "text", "text": f"Full-resolution image: {facts}\nRequest: {prompt.strip()}"},
                ],
            }],
        )
    except anthropic.AuthenticationError as e:
        raise PlannerError("Anthropic credentials missing or invalid (set ANTHROPIC_API_KEY or run `ant auth login`)") from e
    except anthropic.RateLimitError as e:
        raise PlannerError("rate limited by the API; retry shortly") from e
    except anthropic.APIStatusError as e:
        raise PlannerError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise PlannerError(f"could not reach the API: {e}") from e

    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        raise PlannerError(f"the model declined to plan this request (category: {cat or 'unspecified'})")
    if resp.stop_reason == "max_tokens":
        raise PlannerError("plan was truncated (max_tokens); simplify the request")
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise PlannerError("model returned no plan")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlannerError(f"model returned invalid JSON: {e}") from e

    u = resp.usage
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    return raw, usage

