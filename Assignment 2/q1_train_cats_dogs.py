# q1_train_cats_dogs.py
# Train several classifiers on the Cat vs Dog dataset and save the best model.

import os
import glob
import pickle

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# This is your Q1 training folder
DATA_DIR = "Q1/train"
IMAGE_SIZE = (64, 64)
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "cats_dogs_best.pkl")


def discover_classes(data_dir: str):
    """
    Look for subfolders inside data_dir.
    If we find any, we treat each subfolder as a class.
    If not, we fall back to using filenames containing 'cat' or 'dog'.
    """
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    subdirs = sorted(subdirs)

    if subdirs:
        print("Discovered class folders:", subdirs)
        return subdirs, "folders"
    else:
        print("No class folders found in Q1/train.")
        print("Will infer labels from filenames containing 'cat' or 'dog'.")
        return ["cat", "dog"], "filenames"


def load_images(data_dir: str):
    """
    Load images and labels.
    - If using folders: label is folder name.
    - If using filenames: label is from 'cat' / 'dog' in the filename.
    """
    class_names, mode = discover_classes(data_dir)
    X, y = [], []

    if mode == "folders":
        for label_idx, class_name in enumerate(class_names):
            class_folder = os.path.join(data_dir, class_name)
            pattern_list = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            file_list = []
            for pattern in pattern_list:
                file_list.extend(glob.glob(os.path.join(class_folder, pattern)))

            print(f"Found {len(file_list)} images in {class_folder}")

            for img_path in file_list:
                try:
                    img = Image.open(img_path).convert("L")
                    img = img.resize(IMAGE_SIZE)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    X.append(arr.flatten())
                    y.append(label_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    else:
        # All images directly inside Q1/train – label from filename
        pattern_list = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        file_list = []
        for pattern in pattern_list:
            file_list.extend(glob.glob(os.path.join(data_dir, pattern)))

        print(f"Found {len(file_list)} total images in {data_dir}")

        for img_path in file_list:
            fname = os.path.basename(img_path).lower()
            if "cat" in fname:
                label_idx = 0  # 'cat'
            elif "dog" in fname:
                label_idx = 1  # 'dog'
            else:
                # skip files that don't clearly say cat/dog
                continue

            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(IMAGE_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"Total loaded samples: {X.shape[0]}")
    return X, y, class_names


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    X, y, class_names = load_images(DATA_DIR)
    if X.size == 0:
        print("No data loaded. Check your Q1/train folder and file names.")
        return

    # 2. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 3. Models to compare
    models = [
        (
            "logreg",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500))
            ])
        ),
        (
            "svm_rbf",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            probability=True))
            ])
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42
            )
        )
    ]

    best_model = None
    best_name = None
    best_acc = 0.0

    # 4. Train & evaluate
    for name, model in models:
        print(f"\n=== Training model: {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{name} validation accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = (model, class_names)
            best_name = name

    if best_model is None:
        print("No model trained successfully.")
        return

    model_obj, class_names_best = best_model
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    print("\nClassification report for best model:")
    print(classification_report(y_val, model_obj.predict(X_val),
                                target_names=class_names))

    # 5. Save model + class names
    to_save = {"model": model_obj, "class_names": class_names_best}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(to_save, f)
    print(f"\nSaved best model and class names to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
