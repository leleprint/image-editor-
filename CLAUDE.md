# House rules (print shop)

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

## Mockups
- Default placements follow `imgedit/mockup/placements.py`; always state assumptions (garment size, bag size).
- Always keep pixels outside the logo identical to the product photo (verified by the engine).
