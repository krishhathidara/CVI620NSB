# q2_infer_mnist.py
# Load the saved MNIST model and run predictions on random samples.

import os
import pickle

import numpy as np
import pandas as pd

# === CONFIG – adjust file names if needed ===
DATA_PATH = "Q2/mnist_test.csv"   # or mnist_train.csv if you prefer
MODEL_PATH = os.path.join("models", "mnist_best.pkl")



def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}. Train first with q2_train_mnist.py")
        return

    if not os.path.isfile(DATA_PATH):
        print(f"CSV file not found: {DATA_PATH}")
        return

    # 1. Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {MODEL_PATH}")

    # 2. Load some data to test inference
    df = pd.read_csv(DATA_PATH)
    y_true = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0

    print(f"Loaded {X.shape[0]} samples for inference.")

    # 3. Pick a few random samples to display predictions
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=10, replace=False)

    print("\nSample predictions:")
    for idx in indices:
        x = X[idx].reshape(1, -1)
        pred = int(model.predict(x)[0])
        print(f"Index {idx:5d} | True: {y_true[idx]} | Predicted: {pred}")


if __name__ == "__main__":
    main()
