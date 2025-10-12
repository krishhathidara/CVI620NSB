# A1_Q2.py — Part II: Invisible Cloak (Split View with Frozen Background)
# -----------------------------------------------------------------------
# Keys:
#   ENTER → capture clean background (frozen on left)
#   S     → save snapshot (left & right)
#   Q     → quit

import cv2
import numpy as np
from datetime import datetime
import time

# --- Green color range (HSV) ---
LOWER_GREEN_1 = np.array([35, 80, 50])
UPPER_GREEN_1 = np.array([85, 255, 255])
LOWER_GREEN_2 = np.array([25, 80, 50])
UPPER_GREEN_2 = np.array([35, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)

def build_green_mask(hsv):
    """Create and clean green mask using morphologyEx."""
    mask1 = cv2.inRange(hsv, LOWER_GREEN_1, UPPER_GREEN_1)
    mask2 = cv2.inRange(hsv, LOWER_GREEN_2, UPPER_GREEN_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Opening = erosion + dilation (removes small noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    # Slight dilation to ensure full green coverage
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    return mask

def apply_invisible(frame, background):
    """Replace green region with background."""
    if background is None:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_green_mask(hsv)
    inv_mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    bg = cv2.bitwise_and(background, background, mask=mask)
    return cv2.add(fg, bg)

def overlay(img, text, y, color=(255,255,255)):
    """Overlay text with outline."""
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    time.sleep(2)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    background = None
    frozen_bg = None
    print("Controls: ENTER=capture background   S=save snapshot   Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Frame not captured, retrying...")
            continue

        # If frozen_bg is set, use that on left; otherwise live feed
        left_display = frozen_bg if frozen_bg is not None else frame

        # Right side always shows invisibility effect
        invisible = apply_invisible(frame, background)

        split = np.hstack([left_display, invisible])

        h, w = split.shape[:2]
        overlay(split, "Left: Captured Background | Right: Invisible Cloak", 30)
        overlay(split, "Press ENTER to capture background, S to save, Q to quit", h - 20, (0,255,255))

        cv2.imshow("Part II - Invisible Cloak", split)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER = capture background
            print("Capturing clean background... please step out of view.")
            time.sleep(2)
            frames = []
            for i in range(40):
                ok, f = cap.read()
                if ok:
                    frames.append(f.astype(np.float32))
                    cv2.waitKey(10)
            if frames:
                background = np.uint8(np.mean(frames, axis=0))
                frozen_bg = background.copy()  # freeze for left side
                print("✅ Background captured and frozen on left view.")

        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"left_{ts}.png", left_display)
            cv2.imwrite(f"right_{ts}.png", invisible)
            print(f"💾 Saved: left_{ts}.png and right_{ts}.png")

        elif key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
