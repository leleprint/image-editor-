"""Zero-token fast path.

Handles unambiguous, whole-image requests ("rotate 90 clockwise and make it a
bit warmer") without calling the model. It only answers when *every* clause of
the request is understood; otherwise it returns None and the model plans it.
"""

from __future__ import annotations

import re
from typing import Optional

_SPLIT = re.compile(r"\s*(?:,|;|\.(?:\s|$)|\band then\b|\bthen\b|\band\b|\balso\b|\bplus\b)\s*")
_FILLER = re.compile(
    r"\b(please|can you|could you|i want to|i'd like to|i would like to|kindly|just|the (image|photo|picture|pic)|"
    r"this (image|photo|picture|pic)|it|image|photo|picture|overall|a little more|make|look|it's|its)\b"
)
_LESS = r"(slightly|a bit|a little|a touch|subtly|somewhat|lightly|gently|mildly)"
_MORE = r"(much|a lot|significantly|strongly|very|way|heavily|dramatically)"

_ASPECTS = {"square": (1, 1)}


def _intensity(text: str) -> float:
    if re.search(_LESS, text):
        return 0.5
    if re.search(_MORE, text):
        return 1.8
    return 1.0


def _percent(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent)", text)
    return float(m.group(1)) / 100 if m else None


def _clean(text: str) -> str:
    t = text.lower().strip()
    t = _FILLER.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(" .!")


def _step(clause: int, op: str, why: str, **params) -> dict:
    return {"clause": clause, "op": op, "params": [{"name": k, "value": v} for k, v in params.items()],
            "region": None, "why": why}


def _match(i: int, raw: str, size: tuple[int, int], amb: list[dict]) -> Optional[list[dict]]:
    t = _clean(raw)
    k = _intensity(t)
    pct = _percent(t)
    sign_up = re.search(r"\b(increase|boost|more|raise|add|higher|up|enhance|stronger)\b", t)
    sign_dn = re.search(r"\b(decrease|reduce|less|lower|down|tone down|weaker|remove)\b", t)

    def signed(base: float) -> Optional[float]:
        if sign_up and not sign_dn:
            return base
        if sign_dn and not sign_up:
            return -base
        return None

    # --- geometric
    m = re.fullmatch(r"(rotate|turn)( by)? (-?\d+(?:\.\d+)?) ?(°|deg|degrees)?( (clockwise|cw|counter-?clockwise|ccw|anti-?clockwise))?", t)
    if m:
        deg = float(m.group(3))
        d = m.group(6)
        if d is None:
            amb.append({"issue": "rotation direction not stated", "assumption": "clockwise"})
            d = "cw"
        ccw = d.startswith(("counter", "ccw", "anti"))
        return [_step(i, "rotate", "requested rotation", degrees=deg if ccw else -deg)]
    m = re.fullmatch(r"(rotate|turn) (left|right)", t)
    if m:
        return [_step(i, "rotate", "requested rotation", degrees=90.0 if m.group(2) == "left" else -90.0)]
    m = re.fullmatch(r"(flip|mirror)( (horizontally|vertically|left to right|upside down|top to bottom))?", t)
    if m:
        d = m.group(3)
        if d is None:
            amb.append({"issue": "flip axis not stated", "assumption": "horizontal (mirror left-right)"})
        vertical = d in ("vertically", "upside down", "top to bottom")
        return [_step(i, "flip", "requested mirror", direction="vertical" if vertical else "horizontal")]
    m = re.fullmatch(r"(resize|scale)( down| up)? (to|by) (\d+(?:\.\d+)?) ?(%|percent)", t)
    if m:
        return [_step(i, "resize", "requested scale", scale=float(m.group(4)) / 100)]
    m = re.fullmatch(r"(resize|scale)( down| up)? to (\d+) ?[x×] ?(\d+)( ?px| pixels)?", t)
    if m:
        w, h = int(m.group(3)), int(m.group(4))
        sw, sh = size
        if abs(w / h - sw / sh) > 0.01:
            amb.append({"issue": f"{w}x{h} has a different aspect ratio than the source {sw}x{sh}",
                        "assumption": "stretch to exactly the requested size (no crop)"})
        return [_step(i, "resize", "requested size", width=w, height=h)]
    m = re.fullmatch(r"(resize|scale)( down| up)? to (\d+) ?(px|pixels)? (wide|width|tall|high|height)", t)
    if m:
        key = "width" if m.group(5) in ("wide", "width") else "height"
        return [_step(i, "resize", "requested size", **{key: int(m.group(3))})]
    if re.fullmatch(r"(half|halve)( the)? (size|resolution)|(scale|resize) (down )?by half", t):
        return [_step(i, "resize", "requested scale", scale=0.5)]
    if re.fullmatch(r"double( the)? (size|resolution)|(scale|resize) up by 2x?|upscale 2x", t):
        return [_step(i, "resize", "requested scale", scale=2.0)]
    m = re.fullmatch(r"crop( to)?( a)? (square|(\d+) ?[:/x] ?(\d+))( aspect( ratio)?)?( centered| center)?", t)
    if m:
        aw, ah = _ASPECTS.get(m.group(3)) or (int(m.group(4)), int(m.group(5)))
        if aw <= 0 or ah <= 0:
            return None
        sw, sh = size
        target = aw / ah
        if sw / sh > target:
            nw = sh * target / sw
            box = ((1 - nw) / 2, 0.0, (1 + nw) / 2, 1.0)
        else:
            nh = sw / target / sh
            box = (0.0, (1 - nh) / 2, 1.0, (1 + nh) / 2)
        amb.append({"issue": "crop position not stated", "assumption": "centered crop"})
        return [_step(i, "crop", f"centered {aw}:{ah} crop", x0=box[0], y0=box[1], x1=box[2], y1=box[3])]
    m = re.fullmatch(r"add( a)?( (\d+) ?(px|pixels?))?( (white|black|gr[ae]y|#[0-9a-f]{3}(?:[0-9a-f]{3})?))? (border|frame|padding)", t)
    if m:
        px_ = int(m.group(3)) if m.group(3) else max(1, round(min(size) * 0.05))
        if not m.group(3):
            amb.append({"issue": "border width not stated", "assumption": f"{px_}px (5% of short side)"})
        color = m.group(6) or "white"
        return [_step(i, "pad", "requested border", top=px_, right=px_, bottom=px_, left=px_, color=color)]

    # --- whole-image looks
    if re.fullmatch(r"(convert to |turn into )?(black and white|black & white|b&w|b/w|gr[ae]yscale|monochrome)", t):
        return [_step(i, "grayscale", "requested black and white", amount=1.0)]
    if re.fullmatch(r"(convert to |apply |add )?(a )?sepia( tone| effect| filter)?", t):
        return [_step(i, "sepia", "requested sepia", amount=1.0)]
    if re.fullmatch(r"invert( colou?rs)?|(make )?(a )?negative", t):
        return [_step(i, "invert", "requested negative")]
    if re.fullmatch(r"auto ?(contrast|levels)|(fix|stretch) (the )?levels", t):
        return [_step(i, "auto_contrast", "requested automatic levels", clip_percent=0.5)]

    # --- adjustments with direction words
    m = re.fullmatch(r"(?:(?:increase|raise|boost)? ?exposure|expose) (?:by )?([+-]?\d+(?:\.\d+)?) ?(stops?|ev)", t)
    if m:
        return [_step(i, "exposure", "requested exposure change", stops=float(m.group(1)))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(brighten|lighten)( up)?( by \d+(\.\d+)? ?(%|percent))?|"
                    rf"({_LESS} |{_MORE} )?(brighter|lighter)( by \d+(\.\d+)? ?(%|percent))?", t):
        return [_step(i, "brightness", "requested brighter", amount=min(1.0, pct if pct is not None else 0.25 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(darken|darker)( by \d+(\.\d+)? ?(%|percent))?", t):
        return [_step(i, "brightness", "requested darker", amount=-min(1.0, pct if pct is not None else 0.25 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(warmer|warm up)", t):
        return [_step(i, "temperature", "requested warmer", amount=min(1.0, 0.3 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(cooler|colder|cool down)", t):
        return [_step(i, "temperature", "requested cooler", amount=-min(1.0, 0.3 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(sharpen|sharper|crisper)( up)?", t):
        return [_step(i, "sharpen", "requested sharpening", amount=min(3.0, 0.6 * k), radius=1.2, threshold=0.01)]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(blur|blurry|soften|softer)", t):
        return [_step(i, "blur", "requested blur", radius=3.0 * k)]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(denoise|reduce (the )?noise|remove (the )?noise|less noisy|clean up (the )?noise)", t):
        return [_step(i, "denoise", "requested noise reduction", strength=min(1.0, 0.5 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(more )?(vivid|vibrant|colou?rful|punchier|pop)", t):
        return [_step(i, "vibrance", "requested more vivid colour", amount=min(1.0, 0.3 * k))]
    if re.fullmatch(rf"({_LESS} |{_MORE} )?(less )?(saturated|muted)", t) or re.fullmatch(r"(desaturate|mute)( the)?( colou?rs)?", t):
        return [_step(i, "saturation", "requested less saturation", amount=-min(1.0, 0.3 * k))]
    if re.fullmatch(rf"(add )?(a )?({_LESS} |{_MORE} )?vignette", t):
        return [_step(i, "vignette", "requested vignette", amount=-min(1.0, 0.3 * k), radius=0.8)]
    m = re.search(r"\b(brightness|contrast|saturation|vibrance|warmth|temperature|sharpness|exposure)\b", t)
    if m and re.fullmatch(rf"((increase|boost|more|raise|add|higher|enhance|decrease|reduce|less|lower|tone down) )?"
                          rf"({_LESS} |{_MORE} )?(the )?{m.group(1)}( (up|down|higher|lower))?"
                          rf"( by \d+(\.\d+)? ?(%|percent))?", t):
        what = m.group(1)
        base = {"brightness": 0.25, "contrast": 0.25, "saturation": 0.25, "vibrance": 0.3,
                "warmth": 0.3, "temperature": 0.3, "sharpness": 0.6, "exposure": 0.5}[what]
        amount = signed(pct if pct is not None and what != "exposure" else base * k)
        if amount is None:
            return None
        if what == "sharpness":
            return [_step(i, "sharpen", "requested sharpening", amount=min(3.0, abs(amount)), radius=1.2, threshold=0.01)] if amount > 0 else None
        op = {"warmth": "temperature", "exposure": "exposure"}.get(what, what)
        key = "stops" if op == "exposure" else "amount"
        return [_step(i, op, f"requested {what} change", **{key: max(-1.0, min(1.0, amount)) if key == "amount" else amount})]
    return None


def plan_locally(prompt: str, size: tuple[int, int]) -> Optional[dict]:
    protected = re.sub(r"\bblack\s+(and|&)\s+white\b", "b&w", prompt, flags=re.I)
    parts = [p for p in _SPLIT.split(protected) if p and p.strip(" .!")]
    if not parts or len(parts) > 8:
        return None
    steps: list[dict] = []
    amb: list[dict] = []
    for i, part in enumerate(parts):
        got = _match(i, part, size, amb)
        if got is None:
            return None
        steps.extend(got)
    geo = any(s["op"] in ("rotate", "flip", "resize", "crop", "pad") for s in steps)
    tonal = any(s["op"] not in ("rotate", "flip", "resize", "crop", "pad") for s in steps)
    preserved = ["every pixel property not named above"]
    if not geo:
        preserved.insert(0, "dimensions, orientation and framing")
    if not tonal:
        preserved.insert(0, "colours and tones")
    return {
        "understanding": "; ".join(p.strip() for p in parts),
        "clauses": [p.strip() for p in parts],
        "ambiguities": amb,
        "needs_clarification": False,
        "question": "",
        "steps": steps,
        "unsupported": [],
        "preserved": preserved,
        "risks": [],
    }
