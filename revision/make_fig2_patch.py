"""Patch Fig. 2 (paper/images/Data flowchart.png) for Reviewer 1, Comment 7.

Replaces the generic text in the context-feature and static-feature boxes with the actual
dimensions and construction parameters, adds d = 86 to the unified-feature box, and states
that normalization is fitted on the training split. Text areas are cleared by interpolating
each pixel row between the area's left and right edge pixels, which reproduces the boxes'
gradient fill. Writes paper/images/Data_flowchart_rev.png; the original is untouched.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "images", "Data flowchart.png")
DST = os.path.join(ROOT, "paper", "images", "Data_flowchart_rev.png")
INK = (12, 22, 40)


def clear(a, x0, y0, x1, y1):
    """Row-wise linear interpolation between the pixels just outside the area."""
    left = a[y0:y1, x0 - 1].astype(float)
    right = a[y0:y1, x1].astype(float)
    w = np.linspace(0, 1, x1 - x0)[None, :, None]
    a[y0:y1, x0:x1] = (left[:, None, :] * (1 - w) + right[:, None, :] * w).astype(np.uint8)


def clear_v(a, x0, y0, x1, y1):
    """Column-wise linear interpolation between the rows just above and below the area."""
    top = a[y0 - 1, x0:x1].astype(float)
    bot = a[y1, x0:x1].astype(float)
    w = np.linspace(0, 1, y1 - y0)[:, None, None]
    a[y0:y1, x0:x1] = (top[None, :, :] * (1 - w) + bot[None, :, :] * w).astype(np.uint8)


def font(size, bold=True):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered(d, cx, y, text, f):
    w = d.textlength(text, font=f)
    d.text((cx - w / 2, y), text, fill=INK, font=f)


def main():
    a = np.array(Image.open(SRC).convert("RGB"))
    for x0, y0, x1, y1 in [(500, 228, 752, 330), (505, 700, 745, 790)]:
        clear(a, x0, y0, x1, y1)
    clear_v(a, 1030, 540, 1218, 600)
    a[470:505, 1252:1385] = a[520, 1300]  # plain page background below the arrow
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    f = font(19)
    fs = font(17)
    cx = (500 + 752) / 2
    centered(d, cx, 232, "17 causal per-source features", f)
    centered(d, cx, 260, "burst 6, dormant 3,", fs)
    centered(d, cx, 284, "diversity 4, transition 4", fs)
    centered(d, cx, 308, "windows W = 10 s, 60 s", fs)
    cx = (505 + 745) / 2
    centered(d, cx, 704, "69 CICFlowMeter features", f)
    centered(d, cx, 732, "IDs, IPs, time fields removed;", fs)
    centered(d, cx, 756, "constant columns filtered", fs)
    centered(d, (1020 + 1228) / 2, 552, "d = 69 + 17 = 86", font(21))
    centered(d, (1250 + 1385) / 2, 476, "fit on train split", font(15))
    # mild global enhancement (style: bold & bright, Fig. 6 reference); content unchanged
    from PIL import ImageEnhance
    im = ImageEnhance.Brightness(im).enhance(1.04)
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Color(im).enhance(1.15)
    im = ImageEnhance.Sharpness(im).enhance(1.6)
    im.save(DST)
    print("wrote", DST)


if __name__ == "__main__":
    main()
