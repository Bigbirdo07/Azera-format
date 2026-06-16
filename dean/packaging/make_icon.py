#!/usr/bin/env python3
"""Build the Dean app icon (.icns for macOS, .ico for Windows) from the
azera hummingbird artwork — styled like a modern macOS app icon.

Design:
  * The hummingbird is *lifted off* its black background (brightness -> alpha)
    so its neon-cyan glow can bloom over a custom backdrop instead of a flat
    black square.
  * It sits on a rounded "squircle" tile (Apple HIG: ~824px content in a 1024
    canvas, corner radius ~22.4%) with a deep teal vertical gradient and a soft
    radial cyan bloom behind the bird.
  * A subtle drop shadow under the tile gives it depth in the Dock, the way
    Claude / Chrome icons read.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC = Path("/Users/albertopaz/azera-bioanalytics/azera-hummingbird-logo.png")
OUT_DIR = Path(__file__).resolve().parent
MASTER = OUT_DIR / "dean_icon_master.png"
ICNS = OUT_DIR / "Dean.icns"
ICO = OUT_DIR / "Dean.ico"

CANVAS = 1024
TILE = 824                      # rounded tile size within the canvas (HIG-ish)
RADIUS = int(TILE * 0.2237)     # macOS squircle corner radius
BIRD_BBOX = (455, 90, 1335, 715)  # hummingbird bounding box in the 1535x1024 source

# Palette (deep teal tile, neon-cyan bird).
TOP = (15, 64, 71)      # #0F4047  gradient top
BOTTOM = (4, 20, 25)    # #041419  gradient bottom
BLOOM = (11, 224, 200)  # #0BE0C8  cyan bloom behind the bird


def _vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    t = np.linspace(0.0, 1.0, size)[:, None]
    top_a = np.array(top, dtype=float)
    bot_a = np.array(bottom, dtype=float)
    row = top_a[None, :] * (1 - t) + bot_a[None, :] * t          # (size, 3)
    grid = np.repeat(row[:, None, :], size, axis=1)               # (size, size, 3)
    return Image.fromarray(grid.astype("uint8"), "RGB")


def _radial_bloom(size: int, color: tuple, cx: float, cy: float, radius: float, strength: float) -> Image.Image:
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / radius
    intensity = np.clip(1.0 - dist, 0.0, 1.0) ** 2 * strength
    rgba = np.zeros((size, size, 4), dtype=float)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = intensity * 255
    return Image.fromarray(rgba.astype("uint8"), "RGBA")


def _extract_bird() -> Image.Image:
    """Lift the bird off pure black: brightness becomes the alpha channel."""
    bird = Image.open(SRC).convert("RGB").crop(BIRD_BBOX)
    arr = np.asarray(bird).astype(float)
    # Perceived luminance, weighted toward the cyan channels the bird uses.
    lum = 0.15 * arr[..., 0] + 0.55 * arr[..., 1] + 0.30 * arr[..., 2]
    alpha = np.clip((lum / 255.0) ** 0.8 * 1.25, 0.0, 1.0) * 255
    rgba = np.dstack([arr, alpha]).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def build_master() -> Image.Image:
    # --- tile background -------------------------------------------------
    tile = _vertical_gradient(TILE, TOP, BOTTOM).convert("RGBA")
    bloom = _radial_bloom(TILE, BLOOM, cx=TILE * 0.52, cy=TILE * 0.46, radius=TILE * 0.55, strength=0.45)
    tile = Image.alpha_composite(tile, bloom)

    # --- bird (with its own soft glow) ----------------------------------
    bird = _extract_bird()
    target_w = int(TILE * 0.80)
    scale = target_w / bird.width
    bird = bird.resize((target_w, int(bird.height * scale)), Image.LANCZOS)

    glow = bird.filter(ImageFilter.GaussianBlur(TILE * 0.012))
    bird_layer = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    bx = (TILE - bird.width) // 2
    by = int((TILE - bird.height) * 0.46)  # optically centered (slightly high)
    bird_layer.alpha_composite(glow, (bx, by))
    bird_layer.alpha_composite(bird, (bx, by))
    tile = Image.alpha_composite(tile, bird_layer)

    # --- round the tile + thin inner highlight for a glassy edge --------
    tile.putalpha(_rounded_mask(TILE, RADIUS))
    edge = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle(
        [1, 1, TILE - 2, TILE - 2], radius=RADIUS, outline=(120, 230, 220, 60), width=2
    )
    tile = Image.alpha_composite(tile, edge)

    # --- compose on transparent 1024 canvas with a drop shadow ----------
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    off = (CANVAS - TILE) // 2
    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    sh_mask = _rounded_mask(TILE, RADIUS)
    shadow.paste((0, 0, 0, 150), (off, off + 14), sh_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(tile, (off, off))

    canvas.save(MASTER)
    return canvas


def build_icns(master: Image.Image) -> None:
    iconset = OUT_DIR / "Dean.iconset"
    iconset.mkdir(exist_ok=True)
    specs = [
        (16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png"),
    ]
    for size, name in specs:
        master.resize((size, size), Image.LANCZOS).save(iconset / name)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ICNS)], check=True)
    for png in iconset.iterdir():
        png.unlink()
    iconset.rmdir()


def build_ico(master: Image.Image) -> None:
    master.save(ICO, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def main() -> None:
    master = build_master()
    build_icns(master)
    build_ico(master)
    print(f"wrote {ICNS}")
    print(f"wrote {ICO}")
    print(f"preview: {MASTER}")


if __name__ == "__main__":
    main()
