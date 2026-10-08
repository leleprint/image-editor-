# House rules (print shop)

- Rispondi sempre al cliente in **italiano**.

## Production
- Logos with only a few flat colours are produced with **vinyl cutting (HTV)** on a plotter using
  **Siser EasyWeed** (90 micron PU film, sticky carrier); every cut is weeded by hand.
  Lines and gaps must be **>= 1.5 mm** at the printed size (distributor Stahls' guidance for EasyWeed;
  Siser publishes no official minimum). Lower the threshold only if the shop's own test cut proves it.
  The mockup report checks this per colour layer ("ink colour(s) ... -> vinyl (HTV, weeded by hand): OK / TOO THIN").
- When a logo is too thin for vinyl at the requested size, say so in the plan and give the minimum size
  that works, or suggest DTF (min ~0.5 mm). Never thicken the client's master logo silently.

## Client logos
- Dance Studio Bovolone: master logo = `dance-studio-logo.svg` (thin line, unchanged).
  A **bold variant** (`dance-studio-logo-bold.svg`: line x3.2, ring x2.4 inward, same shapes/colours/centring)
  is used **only for mockups**. Bold logo is vinyl-ready from ~225 mm wide (white line governs);
  below that use DTF.
- Text lockup (as printed on the club t-shirts): "DANCE STUDIO / BOVOLONE", two centred lines LEFT of the
  circle, colour = ring pink #EDCFD1 (same vinyl). Font on the shirt is unconfirmed; closest free match is
  URW Gothic Demi (Avant Garde clone). Proportions in circle diameters D: line 1 width 2.50 D, cap 0.263 D,
  baseline pitch 1.21 x cap, gap to circle 0.072 D, text block centre 0.07 D above circle centre.
  Files: dance-studio-logo-text.svg (master) / dance-studio-logo-bold-text.svg (mockups only).

- Teacher names (hoodie back): line 1 "Maestra" in Kaushan Script (closest free match to the old brush
  script), line 2 the name in URW Gothic Demi caps, centred, ring pink; script line 0.75x the name width.
  Script outlines are WELDED (overlapping joins merged) so the plotter cuts each word as one piece.
  Placement: name ~283 mm wide, top ~139 mm below the hood. File: maestra-benedetta.svg.

## Mockups
- Default placements follow `imgedit/mockup/placements.py`; always state assumptions (garment size, bag size).
- Always keep pixels outside the logo identical to the product photo (verified by the engine).

## Final delivery (ALWAYS, at the end of every job)
When a job is finished, without being asked, build ONE folder named after the client
(e.g. "Dance Studio Bovolone") with Italian names, zip it as "<Cliente>.zip" and send it:
  LEGGIMI.txt                 in Italian: what each file is, colours (hex), production notes
                              (vinyl EasyWeed OK / DTF needed, minimum sizes)
  01_Logo/                    official master logo(s): SVG + transparent PNG
  02_Logo_per_mockup/         bold/mockup-only variants (if any)
  03_File_taglio_plotter/     cut-ready SVGs, text converted to outlines, script fonts welded
  04_Mockup/                  final approved mockups only (drop rejected/removed ones)
  05_Font/                    every font file used + its licence file
File names: Italian, lowercase, hyphens (e.g. logo-cerchio.svg, maglietta-fronte.png).
Rebuild the zip after any later change so it always matches the latest approved versions.
