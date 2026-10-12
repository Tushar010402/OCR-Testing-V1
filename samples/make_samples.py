#!/usr/bin/env python3
"""Generate synthetic test images with known text, for smoke-testing the harness.

    python samples/make_samples.py

Writes samples/images/*.png and samples/manifest.csv. The variants cover the
usual OCR pain points: clean print, small text, noise, blur, skew, low contrast.
"""

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TEXTS = {
    "invoice": "INVOICE #INV-2024-0042\nDate: 14 March 2024\nTotal Due: $1,284.50",
    "address": "221B Baker Street\nLondon NW1 6XE\nUnited Kingdom",
    "paragraph": "The quick brown fox jumps over\nthe lazy dog while 7 wizards\nbox and quiz jovial nymphs.",
    "receipt": "Coffee Latte  4.50\nBlueberry Muffin  3.25\nSubtotal  7.75\nTax 8%  0.62",
}


def render(text: str, size: int = 32, fg=(0, 0, 0), bg=(255, 255, 255)) -> Image.Image:
    font = ImageFont.truetype(FONT, size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = probe.multiline_textbbox((0, 0), text, font=font, spacing=size // 3)
    pad = size
    img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), bg)
    ImageDraw.Draw(img).multiline_text((pad - l, pad - t), text, font=font, fill=fg, spacing=size // 3)
    return img


def noisy(img: Image.Image, amount: float = 0.06, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    px = img.load()
    for _ in range(int(img.width * img.height * amount)):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        px[x, y] = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
    return img


VARIANTS = {
    "clean": lambda t: render(t),
    "small": lambda t: render(t, size=12),
    "noisy": lambda t: noisy(render(t)),
    "blurred": lambda t: render(t).filter(ImageFilter.GaussianBlur(1.6)),
    "skewed": lambda t: render(t).rotate(4, expand=True, fillcolor="white"),
    "lowcontrast": lambda t: render(t, fg=(150, 150, 150), bg=(200, 200, 200)),
}


def main():
    out = HERE / "images"
    out.mkdir(exist_ok=True)
    with (HERE / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "ground_truth"])
        for name, text in TEXTS.items():
            for variant, make in VARIANTS.items():
                sid = f"{name}_{variant}"
                make(text).save(out / f"{sid}.png")
                w.writerow([sid, f"images/{sid}.png", text])
    print(f"wrote {len(TEXTS) * len(VARIANTS)} images to {out}")


if __name__ == "__main__":
    main()
