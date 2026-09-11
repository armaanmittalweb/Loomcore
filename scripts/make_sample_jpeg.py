#!/usr/bin/env python3
"""Writes examples/sample.jpg: a procedurally generated (NOT a real photo —
avoids any question of image licensing) test pattern, used only so
`loomcore_example` has something real to JPEG-decode via stb_image by
default. Pass a real photo path as argv[1] to loomcore_example for a
meaningful ImageNet classification instead.
"""
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "examples", "sample.jpg")

if __name__ == "__main__":
    img = Image.new("RGB", (320, 320), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    for i in range(0, 320, 16):
        shade = 60 + (i % 160)
        draw.line([(i, 0), (i, 320)], fill=(shade, 40, 200 - shade % 200), width=6)
    draw.ellipse((60, 60, 260, 260), outline=(240, 220, 40), width=10)
    img.save(OUT, quality=90)
    print(f"wrote {OUT}")
