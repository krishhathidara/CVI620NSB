# OpenCV logo — smaller figure so the bottom wordmark fits comfortably
# Output: opencv_logo_bottom_wordmark.png

import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os

# ===== Canvas & Colors =====
W, H = 1600, 1200
BG = (43, 43, 43)                 # BGR
COL_RED   = (50,  60, 245)        # BGR
COL_GREEN = (100, 215, 120)
COL_BLUE  = (240, 150,  60)

# ===== Ring Geometry (smaller to leave room for text) =====
R_OUT, R_IN = 210, 112            # outer / inner radii (thickness ≈ 98)
OFFSET = 232                      # cluster offset from anchor
X_SPREAD, Y_SPREAD = 0.98, 0.72   # green/blue spread multipliers
NOTCH_HALF_ANGLE = 30             # wedge half-angle (°)

# ===== Wordmark =====
WORD = "OpenCV"
WORDMARK_RATIO = 0.20             # text height ≈ 20% of canvas height
GAP_UNDER_FIGURE = 0.030          # gap from rings bottom to wordmark (fraction of H)
BOTTOM_MARGIN = 0.06              # minimum margin from canvas bottom (fraction of H)

# ---------- helpers ----------
def angle_towards(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

def cut_notch(img, cxy, angle_deg, r_out, r_in, half_angle_deg):
    start = int(angle_deg - half_angle_deg)
    end   = int(angle_deg + half_angle_deg)
    pts_outer = cv2.ellipse2Poly(cxy, (r_out, r_out), 0, start, end, 1)
    cv2.fillConvexPoly(img, np.vstack([[cxy], pts_outer]), BG, cv2.LINE_AA)
    pts_inner = cv2.ellipse2Poly(cxy, (r_in, r_in), 0, start, end, 1)
    cv2.fillConvexPoly(img, np.vstack([[cxy], pts_inner]), BG, cv2.LINE_AA)

def load_font(size: int):
    for p in [
        r"C:\Windows\Fonts\ARLRDBD.TTF",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

# ---------- render ----------
def make_logo() -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)

    # place cluster a bit below center so the red ring never clips
    anchor = np.array([W // 2, int(H * 0.47)])

    red_c   = (int(anchor[0]),                          int(anchor[1] - OFFSET))
    green_c = (int(anchor[0] - int(OFFSET * X_SPREAD)), int(anchor[1] + int(OFFSET * Y_SPREAD)))
    blue_c  = (int(anchor[0] + int(OFFSET * X_SPREAD)), int(anchor[1] + int(OFFSET * Y_SPREAD)))

    for c, col in [(red_c, COL_RED), (green_c, COL_GREEN), (blue_c, COL_BLUE)]:
        cv2.circle(img, c, R_OUT, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, R_IN,  BG,  -1, cv2.LINE_AA)

    centroid = ((red_c[0] + green_c[0] + blue_c[0]) // 3,
                (red_c[1] + green_c[1] + blue_c[1]) // 3)

    cut_notch(img, red_c,   angle_towards(red_c, centroid),   R_OUT+2, R_IN-2, NOTCH_HALF_ANGLE)
    cut_notch(img, green_c, angle_towards(green_c, centroid), R_OUT+2, R_IN-2, NOTCH_HALF_ANGLE)
    cut_notch(img, blue_c,  -90,                               R_OUT+2, R_IN-2, NOTCH_HALF_ANGLE)

    # --- wordmark just below the figure ---
    lowest_ring_y = max(green_c[1] + R_OUT, blue_c[1] + R_OUT)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = load_font(int(H * WORDMARK_RATIO))

    bbox = draw.textbbox((0, 0), WORD, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2

    y = lowest_ring_y + int(GAP_UNDER_FIGURE * H)             # gap under figure
    y = min(y, H - th - int(BOTTOM_MARGIN * H))               # keep bottom margin

    draw.text((x, y), WORD, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    out = "opencv_logo_bottom_wordmark.png"
    cv2.imwrite(out, make_logo())
    print(f"Saved: {out}")
