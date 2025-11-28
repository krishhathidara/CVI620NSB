# q1_infer_cats_dogs.py
# Load saved Cat vs Dog model and run predictions on test images.

import os
import glob
import pickle

import numpy as np
from PIL import Image

IMAGE_SIZE = (64, 64)
MODEL_PATH = os.path.join("models", "cats_dogs_best.pkl")

# Root folder containing test images; can have subfolders (Cat, Dog, etc.)
TEST_ROOT = "Q1/test"


def preprocess_image(path: str):
    """Load and preprocess a single image file into a 1D feature vector."""
    img = Image.open(path).convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)


def list_all_images(root: str):
    """Return a list of image paths (jpg/png/bmp) under root, searching recursively."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        # ** recursive=True ensures we go into Cat/, Dog/, etc. **
        paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return paths


def main():
    # 1. Load model
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}. Train first with q1_train_cats_dogs.py")
        return

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    # New format from q1_train_cats_dogs.py: {"model": ..., "class_names": [...]}
    if isinstance(saved, dict) and "model" in saved and "class_names" in saved:
        model = saved["model"]
        class_names = saved["class_names"]
    else:
        model = saved
        class_names = ["Cat", "Dog"]

    # 2. Find test images
    if not os.path.isdir(TEST_ROOT):
        print(f"Test directory not found: {TEST_ROOT}")
        return

    image_files = list_all_images(TEST_ROOT)

    if not image_files:
        print(f"No images found under {TEST_ROOT}.")
        print("Make sure you have images in subfolders like Q1/test/Cat and Q1/test/Dog.")
        return

    print(f"Found {len(image_files)} images under {TEST_ROOT}. Running predictions...\n")

    # 3. Predict each image
    for img_path in sorted(image_files):
        try:
            X = preprocess_image(img_path)
            pred_idx = int(model.predict(X)[0])

            if 0 <= pred_idx < len(class_names):
                label = class_names[pred_idx]
            else:
                label = f"class_{pred_idx}"

            rel_path = os.path.relpath(img_path, TEST_ROOT)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0][pred_idx]
                print(f"{rel_path} -> {label} (confidence: {proba:.3f})")
            else:
                print(f"{rel_path} -> {label}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
