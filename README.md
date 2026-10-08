# imgedit

A plan-first, scope-locked image editor driven by natural language.

```
imgedit edit photo.jpg "make the sky a bit bluer and brighten the sign on the left"
```

1. **Understand.** The request is split into atomic clauses. Each clause is either mapped to concrete operations or listed as *not done*, with the reason and the closest thing that is possible.
2. **Plan and state limits.** You see every step: the operation, its parameters, the exact region it applies to, any assumptions made, and what will stay unchanged.
3. **Measure risks before writing.** The plan runs once at full resolution in memory. The report then shows measured problems (new clipping, histogram combing/banding, selections that are empty or too broad, upscale factor) and possible artifacts for each step, rated low or elevated from the parameters.
4. **Confirm, then write.** Nothing touches disk until you confirm (or pass `--yes`). The input is never overwritten unless you pass `--in-place`.
5. **Verify the scope.** Every pixel outside the planned regions is copied byte-for-byte from the source. This is then checked; if the check fails, nothing is written.

## Install

```
pip install -e .[llm]        # numpy + pillow, plus the Anthropic SDK for the model planner
export ANTHROPIC_API_KEY=...  # or `ant auth login`
```

## Usage

```
imgedit edit in.jpg "rotate 90 clockwise and make it slightly warmer"   # local rules, 0 tokens
imgedit edit in.jpg "darken the background behind the person" --plan-only
imgedit edit in.jpg "pixelate the licence plate" -o out.png --save-plan plate.json
imgedit apply other.jpg plate.json -o other.out.jpg --yes                # replay, 0 tokens
imgedit ops                                                               # list operations
```

| Flag | Meaning |
|---|---|
| `--plan-only` | show the plan and risk report, write nothing |
| `-y/--yes` | skip the confirmation prompt |
| `--offline` | never call the model; fail if local rules can't parse the request |
| `--force-model` | skip the local fast path |
| `--model`, `--effort` | planner model (default `claude-opus-5-5`) and effort (default `medium`) |
| `--preview-size N` | long side of the preview image sent to the model (default 768) |
| `--dither` | dither edited pixels to hide banding from strong tonal edits |
| `--quality N` | JPEG/WebP quality (default: max(92, estimated source quality); WebP lossless) |
| `--json` | machine-readable report |

Exit codes: `0` success, `1` error, `2` needs clarification (non-interactive), `3` nothing written.

## Token efficiency

The model is called at most once per request. Nothing about the image is sent back and forth.

- **Local fast path.** Simple whole-image requests (rotate, flip, crop to aspect, resize, B&W, sepia, brighter, warmer, contrast, sharpen and so on) are parsed locally at no token cost. This only happens when *every* clause is understood; partial matches go to the model.
- **Plan cache.** Plans are cached on disk, keyed on the preview, the prompt, the model and the effort setting (`~/.cache/imgedit`, override with `IMGEDIT_CACHE`). A rerun costs nothing.
- **Small inputs.** The model gets a downscaled JPEG preview (about 500–800 image tokens at 768px) and a one-line numeric summary of the full-resolution image: size, percentiles and clipping.
- **Cacheable prefix.** The system prompt and operation catalog are frozen and generated deterministically, and marked with `cache_control`. Caching only applies once the prefix is above the model's minimum cacheable length.
- **Terse output.** The plan comes back as schema-constrained JSON (`output_config.format`) with no prose. `--effort low` cuts it further.
- **Free reuse.** `--save-plan` / `apply` reuse a plan on any number of images without calling the model again.

## Guarantees and limits

**Guaranteed**
- No operation runs unless it traces back to a clause of your request. The validator rejects steps that don't, and lists clauses that have no step.
- Out-of-range values are clamped and reported, never changed silently. Invalid steps are dropped and reported.
- Pixels outside the edit regions are byte-identical to the source. After a crop, rotate or resize, they match the source transformed the same way.
- EXIF (orientation reset to 1 after it is applied to the pixels), the ICC profile, DPI, alpha, JPEG chroma subsampling and grayscale mode are kept.
- 90°/180°/270° rotation, flip, crop and pad are lossless. Edits run in float32 and are quantized once.

**Not possible (stated in the plan when asked)**
- Generative edits: removing, adding or moving objects, replacing backgrounds, retouching faces, recovering detail.
- Semantic selection. Regions are rectangles, ellipses, polygons and colour ranges estimated from the preview, so boundaries can be off by a few percent; feathering hides small errors.

**Artifacts the report warns about**
- clipping
- banding
- halos from sharpening
- noise amplification when lifting shadows
- seams from hard region edges
- softness from upscaling or non-right-angle rotation
- JPEG generational loss
- transparency flattened when saving to JPEG
- high-bit-depth or CMYK sources being converted

## Development

```
pip install -e .[llm,dev]
pytest
```
