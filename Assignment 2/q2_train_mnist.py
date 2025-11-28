# q2_train_mnist.py
# Train multiple classifiers on the MNIST CSV dataset and save the best model.

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Use the training CSV that you have in the Q2 folder
DATA_PATH = "Q2/mnist_train.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_best.pkl")

# For speed: we don't need all 40k+ training samples to reach 90% accuracy.
# We'll use at most this many for fitting the models.
MAX_TRAIN_SAMPLES = 20000


def load_mnist(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    print(f"Loaded MNIST CSV with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"CSV file not found: {DATA_PATH}")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load full data
    X, y = load_mnist(DATA_PATH)

    # 2. Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Total train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 3. Optionally sub-sample the training set for faster training
    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        from sklearn.model_selection import train_test_split as tts
        X_train_sub, _, y_train_sub, _ = tts(
            X_train, y_train,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=y_train,
            random_state=42
        )
        print(f"Using a subset of {X_train_sub.shape[0]} samples for training (for speed).")
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # 4. Define models
    models = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1
            )
        ),
        (
            "mlp",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    max_iter=30,
                    random_state=42
                ))
            ])
        )
    ]

    best_model = None
    best_name = None
    best_acc = 0.0

    # 5. Train & evaluate
    for name, model in models:
        print(f"\n=== Training model: {name} ===")
        model.fit(X_train_sub, y_train_sub)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{name} validation accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    if best_model is None:
        print("No model was trained successfully.")
        return

    print(f"\nBest model: {best_name} with validation accuracy {best_acc:.4f}")
    print("\nClassification report for best model:")
    print(classification_report(y_val, best_model.predict(X_val)))

    if best_acc < 0.90:
        print("\n⚠️ Accuracy is below 90%. "
              "If you have time, you can increase MAX_TRAIN_SAMPLES or "
              "tune the models (more estimators, more epochs, etc.).")

    # 6. Save best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
