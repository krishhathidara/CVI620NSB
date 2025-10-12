# A1_Q3.py — Part III: Mini Photo Editor (OpenCV + NumPy + Matplotlib)
# Looks for 'first.jpg' automatically (case-insensitive). If not found/decodable, prompts for a path.
# Menu:
# 1) Brightness  2) Contrast  3) Grayscale  4) Padding (ratio + border type)
# 5) Thresholding (binary / inverse)  6) Blend with another image
# 7) Undo  8) View history  9) Save & Exit

import os
import sys
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Helpers ----------------------------
def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def imread_safely(path: str):
    """Read image from path (supports quotes); returns None if not found or invalid."""
    if not path:
        return None
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  OpenCV couldn't decode this file (unsupported/corrupt?): {path}")
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_side_by_side(before, after, title="Preview"):
    def to_rgb(im):
        if im.ndim == 3:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    fig = plt.figure(figsize=(10, 5))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.axis('off')
    ax1.imshow(to_rgb(before), cmap='gray' if before.ndim == 2 else None)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Preview")
    ax2.axis('off')
    ax2.imshow(to_rgb(after), cmap='gray' if after.ndim == 2 else None)

    plt.tight_layout()
    plt.show()

def ask_int(prompt, default=None, min_val=None, max_val=None):
    while True:
        raw = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            val = default
        else:
            try:
                val = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_val is not None and val < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return val

def ask_float(prompt, default=None, min_val=None, max_val=None):
    while True:
        raw = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            val = default
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if min_val is not None and val < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return val

def pick_from(prompt, options_dict, default_key=None):
    keys = list(options_dict.keys())
    for i, k in enumerate(keys, 1):
        print(f"{i}. {k}")
    while True:
        raw = input(f"{prompt}{' ['+str(default_key)+']' if default_key else ''}: ").strip()
        if not raw and default_key and default_key in options_dict:
            return default_key
        try:
            idx = int(raw)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
        except ValueError:
            if raw in options_dict:
                return raw
        print("Invalid choice. Try again.")

# ---------------------------- Ops ----------------------------
def op_brightness(img, history_log):
    offset = ask_int("Enter brightness offset (negative=darken, positive=brighten)", default=30, min_val=-255, max_val=255)
    out = img.astype(np.int16) + offset
    out = ensure_uint8(out)
    history_log.append(f"brightness {offset:+d}")
    return out

def op_contrast(img, history_log):
    alpha = ask_float("Enter contrast factor (1.0=same, 1.2=stronger, 0.8=weaker)", default=1.2, min_val=0.0, max_val=5.0)
    out = img.astype(np.float32) * alpha
    out = ensure_uint8(out)
    history_log.append(f"contrast x{alpha:.2f}")
    return out

def op_grayscale(img, history_log):
    out = to_gray(img)
    history_log.append("convert to grayscale")
    return out

def compute_padding_to_ratio(h, w, target_w, target_h):
    if target_w <= 0 or target_h <= 0:
        return 0, 0, 0, 0
    r_target = target_w / target_h
    r_current = w / h
    if r_current < r_target:
        new_w = math.ceil(r_target * h)
        pad_w = new_w - w
        left = pad_w // 2
        right = pad_w - left
        top = bottom = 0
    elif r_current > r_target:
        new_h = math.ceil(w / r_target)
        pad_h = new_h - h
        top = pad_h // 2
        bottom = pad_h - top
        left = right = 0
    else:
        top = bottom = left = right = 0
    return top, bottom, left, right

def scale_padding_proportionally(pads, factor):
    top, bottom, left, right = pads
    if factor <= 0:
        return pads
    mul = int(factor) + 1
    return top * mul, bottom * mul, left * mul, right * mul

def op_padding(img, history_log):
    border_types = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP
    }
    print("\nChoose border type:")
    bkey = pick_from("Border type", border_types, default_key="reflect")

    print("\nAspect ratio choices:")
    ratio_presets = {
        "Square (1:1)": (1, 1),
        "Landscape (16:9)": (16, 9),
        "Portrait (9:16)": (9, 16),
        "Custom": None
    }
    rkey = pick_from("Aspect ratio", ratio_presets, default_key="Square (1:1)")

    if rkey == "Custom":
        w_ratio = ask_int("Enter ratio width (W in W:H)", min_val=1, default=4)
        h_ratio = ask_int("Enter ratio height (H in W:H)", min_val=1, default=5)
        target_ratio = (w_ratio, h_ratio)
    else:
        target_ratio = ratio_presets[rkey]

    h, w = img.shape[:2]
    top, bottom, left, right = compute_padding_to_ratio(h, w, target_ratio[0], target_ratio[1])

    print(f"Minimum padding to reach {target_ratio[0]}:{target_ratio[1]} → top={top}, bottom={bottom}, left={left}, right={right}")
    extra_factor = ask_int("Extra padding factor (0=min, 1=2x, 2=3x, ...)", default=0, min_val=0, max_val=20)
    top, bottom, left, right = scale_padding_proportionally((top, bottom, left, right), extra_factor)

    if top == bottom == left == right == 0:
        print("Image already matches the chosen ratio. You can add custom uniform padding instead.")
        uniform = ask_int("Uniform padding (pixels added to all sides)", default=0, min_val=0, max_val=2000)
        top = bottom = left = right = uniform

    if bkey == "constant":
        if img.ndim == 2:
            c = ask_int("Constant gray color (0..255)", default=0, min_val=0, max_val=255)
            border_color = c
        else:
            r = ask_int("Border color RED (0..255)", default=0, min_val=0, max_val=255)
            g = ask_int("Border color GREEN (0..255)", default=0, min_val=0, max_val=255)
            b = ask_int("Border color BLUE (0..255)", default=0, min_val=0, max_val=255)
            border_color = (b, g, r)
        out = cv2.copyMakeBorder(img, top, bottom, left, right, border_types[bkey], value=border_color)
        color_desc = border_color if isinstance(border_color, int) else f"BGR{border_color}"
        history_log.append(f"padded t{top} b{bottom} l{left} r{right} with {bkey} ({color_desc}); ratio {target_ratio[0]}:{target_ratio[1]}; factor {extra_factor}")
    else:
        out = cv2.copyMakeBorder(img, top, bottom, left, right, border_types[bkey])
        history_log.append(f"padded t{top} b{bottom} l{left} r{right} with {bkey}; ratio {target_ratio[0]}:{target_ratio[1]}; factor {extra_factor}")
    return out

def op_threshold(img, history_log):
    mode_options = {"Binary": cv2.THRESH_BINARY, "Binary Inverse": cv2.THRESH_BINARY_INV}
    print("\nThreshold mode:")
    mkey = pick_from("Choose", mode_options, default_key="Binary")

    use_otsu = input("Use Otsu automatic threshold? [y/N]: ").strip().lower() == "y"
    gray = to_gray(img)

    if use_otsu:
        ret, out = cv2.threshold(gray, 0, 255, mode_options[mkey] | cv2.THRESH_OTSU)
        history_log.append(f"threshold {mkey} (Otsu, T≈{ret:.1f})")
    else:
        t = ask_int("Enter threshold value (0..255)", default=128, min_val=0, max_val=255)
        _, out = cv2.threshold(gray, t, 255, mode_options[mkey])
        history_log.append(f"threshold {mkey} (T={t})")
    return out

def op_blend(img, history_log):
    path = input('Enter path to second image (e.g., "C:\\\\Users\\\\me\\\\img.jpg"): ').strip()
    img2 = imread_safely(path)
    if img2 is None:
        print("❌ Could not load second image. Aborting blend.")
        return img

    base = img
    if base.ndim == 2 and img2.ndim == 3:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.ndim == 3 and img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h, w = base.shape[:2]
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = ask_float("Enter alpha (0..1) where result = alpha*img + (1-alpha)*img2", default=0.6, min_val=0.0, max_val=1.0)
    out = cv2.addWeighted(base, alpha, img2_resized, 1 - alpha, 0.0)
    out = ensure_uint8(out)
    history_log.append(f"blend alpha={alpha:.2f} with '{os.path.basename(path)}'")
    return out

# ---------------------------- Loader with diagnostics ----------------------------
def find_first_jpg_case_insensitive(basename="first.jpg"):
    candidates = []
    # exact CWD
    candidates.append(os.path.join(os.getcwd(), basename))
    # case-insensitive in CWD
    candidates += glob.glob(os.path.join(os.getcwd(), "first.*"))
    # next to script
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, basename))
        candidates += glob.glob(os.path.join(here, "first.*"))
    except NameError:
        pass
    # dedup & keep only existing files
    seen = set(); uniq = []
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c) and c.lower() not in seen:
            seen.add(c.lower()); uniq.append(c)
    # prefer .jpg/.jpeg
    uniq.sort(key=lambda p: (os.path.splitext(p)[1].lower() not in (".jpg", ".jpeg"), p))
    return uniq[0] if uniq else None

# ---------------------------- Main ----------------------------
def main():
    print("==== Mini Photo Editor ====\n")
    print(f"📂 Working directory: {os.getcwd()}")
    print("🔎 Looking for 'first.jpg' (any case) here and next to the script...")

    found = find_first_jpg_case_insensitive("first.jpg")
    img = None
    if found:
        print(f"✅ Found candidate: {found}")
        img = imread_safely(found)
        if img is None:
            print("❌ Found the file but could not decode it. I will prompt for another path.")
    else:
        nearby = glob.glob(os.path.join(os.getcwd(), "*.[Jj][Pp][Gg]")) + \
                 glob.glob(os.path.join(os.getcwd(), "*.[Jj][Pp][Ee][Gg]")) + \
                 glob.glob(os.path.join(os.getcwd(), "*.[Pp][Nn][Gg]"))
        if nearby:
            print("🖼️ Images in this folder (first 10):")
            for p in nearby[:10]:
                print("   -", os.path.basename(p))
            if len(nearby) > 10:
                print("   ...")

    while img is None:
        path = input('Enter image path to load (tip: drag the file here from Explorer): ').strip()
        img = imread_safely(path)
        if img is None:
            print("❌ Not found or unreadable. Tips:")
            print("   • Confirm the path exists (try: dir \"<that path>\" in PowerShell)")
            print("   • Watch for FIRST.JPG vs first.jpg or hidden .jpg.jpg double extensions")
            print("   • Remove extra quotes/spaces around the path")

    current = ensure_uint8(img.copy())
    history_stack = [current.copy()]
    history_log = ["loaded image"]

    ops = {
        "1": ("Adjust Brightness", op_brightness),
        "2": ("Adjust Contrast", op_contrast),
        "3": ("Convert to Grayscale", op_grayscale),
        "4": ("Add Padding (border type + ratio)", op_padding),
        "5": ("Apply Thresholding (binary / inverse)", op_threshold),
        "6": ("Blend with Another Image", op_blend),
        "7": ("Undo Last Operation", None),
        "8": ("View History of Operations", None),
        "9": ("Save and Exit", None),
    }

    while True:
        print("\n==== Mini Photo Editor ====")
        for k in sorted(ops.keys(), key=int):
            print(f"{k}. {ops[k][0]}")
        choice = input("Enter choice (1–9): ").strip()

        if choice not in ops:
            print("Invalid choice. Try again.")
            continue

        if choice in {"1","2","3","4","5","6"}:
            op_name, fn = ops[choice]
            try:
                preview = fn(current.copy(), history_log)
                show_side_by_side(current, preview, title=op_name)
                current = ensure_uint8(preview)
                history_stack.append(current.copy())
            except Exception as e:
                print(f"⚠️ Operation failed: {e}")

        elif choice == "7":  # Undo
            if len(history_stack) > 1:
                history_stack.pop()
                current = history_stack[-1].copy()
                history_log.append("undo last operation")
                print("↩️  Undid last operation.")
                show_side_by_side(current, current, title="Undo: back to previous state")
            else:
                print("Nothing to undo.")

        elif choice == "8":  # View history
            print("\n=== History of Operations ===")
            for i, h in enumerate(history_log, 1):
                print(f"{i}. {h}")

        elif choice == "9":  # Save & Exit
            ans = input("Do you want to save the final image? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                default_name = "edited_first.jpg"
                out_name = input(f"Enter filename to save [{default_name}]: ").strip() or default_name
                ok = cv2.imwrite(out_name, ensure_uint8(current))
                if ok:
                    print(f"💾 Saved: {os.path.abspath(out_name)}")
                else:
                    print("❌ Failed to save file (check path/permissions).")
            print("Goodbye!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited.")
