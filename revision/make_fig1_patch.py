"""Patch Fig. 1 (paper/images/CON-PICTURE2.PNG) for the revision (Reviewer 1, Comment 1).

* Replaces the QPSO panel, which drew Q=1..8 as swarm nodes and stated "Q=6", with a panel
  showing what Q actually does: particle = d-dim amplitude vector (size independent of Q);
  Q sets init perturbation, rotation bound pi/(2Q) and collapse prob 2^-Q; Measure_k -> mask.
* Replaces the "Source/Destination" bullet in F_s (IPs are no longer features).
Writes paper/images/CON-PICTURE2_rev.png; the original file is left untouched.
"""
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "images", "CON-PICTURE2.PNG")
DST = os.path.join(ROOT, "paper", "images", "CON-PICTURE2_rev.png")

PANEL = (10, 537, 640, 829)  # x0, y0, x1, y1 in the 1672x941 original (purple bar kept below)
ARROW_Y = 680
PURPLE, LAV, INK = "#5b2a86", "#f3eefa", "#1f1f3a"


def render_panel(w_px, h_px, scale=3):
    dpi = 100
    fig = plt.figure(figsize=(w_px * scale / dpi, h_px * scale / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")
    fs = 11.0 * scale

    boxes = [(6, 6, 196, h_px - 12), (222, 6, 190, h_px - 12), (438, 6, w_px - 444, h_px - 12)]
    titles = ["Particle (any $Q$)", "$Q$ sets the search", "Mask $\\mathbf{m}^*$"]
    for (x, y, w, h), t in zip(boxes, titles):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                    fc=LAV, ec=PURPLE, lw=2.4 * scale / 2))
        ax.text(x + w / 2, y + 20, t, ha="center", va="center", fontsize=fs * 1.05,
                weight="bold", color=INK)
    for x0, x1 in [(204, 220), (414, 436)]:
        ax.add_patch(FancyArrowPatch((x0, h_px / 2), (x1, h_px / 2), arrowstyle="-|>",
                                     mutation_scale=14 * scale, lw=2.5 * scale / 2, color=PURPLE))

    # Box 1: amplitude vector
    rng = np.random.default_rng(3)
    a = rng.gamma(0.6, 1.0, 46)
    a = a / a.max()
    x0, base, bw = 18, 190, 172 / 46
    for j, v in enumerate(a):
        ax.add_patch(plt.Rectangle((x0 + j * bw, base - 120 * v), bw * 0.8, 120 * v,
                                   fc="#8e6cc0", ec="none"))
    ax.plot([x0, x0 + 172], [base, base], color=INK, lw=1.0 * scale / 2)
    ax.text(104, 220, "$\\mathbf{a}_i\\in\\mathbb{R}^{d}$, $d=86$", ha="center", fontsize=fs, color=INK, weight="bold")
    ax.text(104, 256, "size independent of $Q$", ha="center", fontsize=fs * 0.9,
            color=PURPLE, weight="bold")

    # Box 2: the three Q entry points
    lines = [("init. noise", "$\\rho(Q)=2^{Q/2}/(Q^2+1)$"),
             ("rotation", "$\\phi\\leq\\pi/(2Q)$"),
             ("collapse", "prob. $2^{-Q}$")]
    for n, (lab, eq) in enumerate(lines):
        y = 62 + n * 50
        ax.text(232, y, f"{n + 1}. {lab}", fontsize=fs, color=INK, weight="bold")
        ax.text(250, y + 24, eq, fontsize=fs, color=INK, weight="bold")
    ax.add_patch(FancyArrowPatch((236, 236), (400, 236), arrowstyle="<|-|>",
                                 mutation_scale=10 * scale, lw=1.5 * scale / 2, color=PURPLE))
    ax.text(236, 262, "explore", fontsize=fs * 0.92, color=INK, ha="left", weight="bold")
    ax.text(400, 262, "exploit", fontsize=fs * 0.92, color=INK, ha="right", weight="bold")

    # Box 3: k-of-d measurement grid (86 cells, 40 selected)
    sel = set(np.random.default_rng(7).choice(86, 40, replace=False).tolist())
    gx, gy, c = 452, 42, 11.2
    for j in range(86):
        r, q = divmod(j, 15)
        ax.add_patch(plt.Rectangle((gx + q * c, gy + r * c), c * 0.86, c * 0.86,
                                   fc="#2e9e5b" if j in sel else "white", ec="#7b6a99",
                                   lw=0.6 * scale / 2))
    tx = 452 + 7.5 * c
    for n, t in enumerate(["$k=40$ of $d$,", "$P(j)\\propto a_{ij}^2$",
                           "best on $\\mathcal{D}_{\\rm tr}$"]):
        ax.text(tx, 160 + n * 40, t, ha="center", fontsize=fs, color=INK,
                weight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w_px, h_px), Image.LANCZOS)


def main():
    im = Image.open(SRC).convert("RGB")
    x0, y0, x1, y1 = PANEL
    im.paste(render_panel(x1 - x0, y1 - y0), (x0, y0))
    # remove the leftover hand graphic between the panel and the LSTM box, redraw the arrow
    d0 = ImageDraw.Draw(im)
    d0.rectangle((641, 600, 683, 780), fill=(255, 255, 255))
    d0.line((641, ARROW_Y, 676, ARROW_Y), fill=(0, 0, 0), width=3)
    d0.polygon([(676, ARROW_Y - 7), (684, ARROW_Y), (676, ARROW_Y + 7)], fill=(0, 0, 0))

    # F_s bullet: "Source/Destination" -> "Ports / Protocol"
    d = ImageDraw.Draw(im)
    bg = im.getpixel((1100, 150))
    d.rectangle((958, 144, 1125, 170), fill=bg)
    try:
        font = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    d.text((963, 145), "Ports / Protocol", fill=(20, 20, 20), font=font)

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
