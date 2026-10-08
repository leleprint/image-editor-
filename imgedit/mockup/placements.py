"""Industry placement standards for decorated products.

Collected from printer/merch guides (Oct 2026); figures disagree between
suppliers, so each entry is a range and the planner must state which value
it chose. Sources:
  apparel  https://www.vistaprint.com/hub/t-shirt-design-placement-guide
           https://printful.com/blog/t-shirt-design-placement-guide
           https://www.shopify.com/blog/logo-placement-on-shirt
           https://v2.4over4.com/guide/t-shirt-print-size-guide-how-big-should-your-design-be
  caps     https://www.printful.com/blog/hat-logo-size-and-placement-guide
           https://faq.oregonscreen.com/knowledge-base/how-big-can-my-art-be-on-a-hat
  mugs     https://titanjet.co.za/11oz-white-ceramic-standard-mug/  (210x96mm art, 25mm handle margin)
           https://help.gearment.com/en-us/article/mug-products-vsa3nr
  totes    https://madegooddesigns.com/tote-bag-design/
           https://www.bristol.ac.uk/print-services/departmental-services/merchandise/tote-bags/

All lengths in millimetres. "top_from" names the physical anchor the
vertical offset is measured from.
"""

from __future__ import annotations

IN = 25.4

PLACEMENTS: dict[str, dict] = {
    "tshirt.full_front": dict(width=(10 * IN, 12 * IN), top_offset=(2 * IN, 3 * IN), top_from="collar seam (front neckline)",
                              horizontal="centred on the shirt", note="adult sizes; Shopify allows up to 16in and 3-4in below collar"),
    "tshirt.center_chest": dict(width=(6 * IN, 10 * IN), top_offset=(3 * IN, 5 * IN), top_from="collar seam",
                                horizontal="centred"),
    "tshirt.left_chest": dict(width=(3.5 * IN, 4 * IN), top_offset=(3 * IN, 4 * IN), top_from="shoulder seam / high point shoulder",
                              horizontal="logo centre ~4in from the shirt centre line, on the WEARER's left (viewer's right)",
                              note="Vistaprint measures 7-9in down from the shoulder point to the logo centre"),
    "tshirt.youth_front": dict(width=(7 * IN, 9 * IN), top_offset=(2 * IN, 3 * IN), top_from="collar seam", horizontal="centred"),
    "tshirt.back_yoke": dict(width=(2 * IN, 4 * IN), top_offset=(1 * IN, 2 * IN), top_from="back collar seam", horizontal="centred"),
    "tshirt.full_back": dict(width=(10 * IN, 12 * IN), top_offset=(3 * IN, 4 * IN), top_from="back collar seam", horizontal="centred"),
    "hoodie.front": dict(width=(8 * IN, 11 * IN), top_offset=(3 * IN, 4 * IN), top_from="neckline / hood seam",
                         horizontal="centred", note="must end above the kangaroo pocket"),
    "hoodie.left_chest": dict(width=(3 * IN, 4 * IN), top_offset=(3 * IN, 4 * IN), top_from="shoulder seam",
                              horizontal="wearer's left"),
    "cap.front": dict(width=(3.5 * IN, 5 * IN), height_max=(2 * IN, 2.25 * IN), top_offset=(0.25 * IN, 0.5 * IN),
                      top_from="top of front panels", horizontal="centred on the front seam",
                      note="unstructured/dad caps: 3.5-4in wide, max 1.75-2in tall; logo bottom sits ~0.5in above the visor seam"),
    "cap.side": dict(width=(1.5 * IN, 2.25 * IN), height_max=(0.75 * IN, 1 * IN), top_offset=(0, 0), top_from="side panel centre",
                     horizontal="centred on side panel"),
    "mug11.front": dict(width=(60, 90), height_max=(80, 96), wrap_width=210, top_offset=(10, 20), top_from="rim",
                        horizontal="centred on the side facing the viewer, opposite or beside the handle, keeping 25mm from the handle",
                        note="11oz mug: ~82mm diameter, ~95mm tall; wrap print area 210x96mm"),
    "mug15.front": dict(width=(70, 100), height_max=(90, 110), wrap_width=230, top_offset=(10, 20), top_from="rim",
                        horizontal="centred, 25mm from the handle", note="15oz mug: ~87mm diameter, ~117mm tall"),
    "tote.front": dict(width=(10 * IN, 12 * IN), top_offset=(2 * IN, 3 * IN), top_from="bag opening (top edge)",
                       horizontal="centred", note="standard print areas 250x250mm to 297x350mm; typical bag 380x420mm"),
    "bottle.front": dict(width=(40, 70), top_offset=(30, 60), top_from="shoulder of the bottle", horizontal="centred on the visible face"),
    "pen.barrel": dict(width=(30, 45), height_max=(6, 8), top_offset=(0, 0), top_from="barrel centre line", horizontal="centred on barrel"),
    "box.face": dict(width=None, top_offset=None, top_from="face edges", horizontal="centred unless the request says otherwise"),
    "card.front": dict(width=None, top_offset=None, top_from="card edges", horizontal="per request; keep 3mm safe margin"),
}

# Typical physical sizes to convert pixels to millimetres from a visible reference.
REFERENCE_SIZES = {
    "tshirt M chest width (armpit to armpit, flat)": 510,
    "tshirt L chest width": 560,
    "tshirt M body length (high point shoulder to hem)": 720,
    "hoodie M chest width": 560,
    "cap front panel height (crown to visor seam, structured)": 105,
    "cap visor width": 190,
    "mug 11oz diameter": 82,
    "mug 11oz height": 95,
    "mug 15oz diameter": 87,
    "mug 15oz height": 117,
    "tote bag width": 380,
    "tote bag height": 420,
    "water bottle 750ml diameter": 73,
    "A4 width": 210,
    "business card width (EU)": 85,
    "business card width (US)": 88.9,
}


def placement_text() -> str:
    """Compact table for the planner prompt (deterministic for prompt caching)."""
    lines = []
    for k, p in PLACEMENTS.items():
        parts = [k]
        if p.get("width"):
            parts.append(f"w {p['width'][0]:.0f}-{p['width'][1]:.0f}mm")
        if p.get("height_max"):
            parts.append(f"h<= {p['height_max'][1]:.0f}mm")
        if p.get("top_offset"):
            parts.append(f"top {p['top_offset'][0]:.0f}-{p['top_offset'][1]:.0f}mm below {p['top_from']}")
        parts.append(p["horizontal"])
        if p.get("note"):
            parts.append(p["note"])
        lines.append("; ".join(parts))
    lines.append("reference sizes (mm): " + "; ".join(f"{k}={v:g}" for k, v in REFERENCE_SIZES.items()))
    return "\n".join(lines)
