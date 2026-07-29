"""
Lesson 1 — Load and explore the MNIST dataset

Concepts: dataset, features (X), labels (y), train/test split, pixel
representation of an image, why we look at data before modeling anything.

Read the TODOs top to bottom. Run the script after each one to check your
work — don't wait until the end.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load MNIST. as_frame=False gives raw NumPy arrays instead of a
    # pandas DataFrame. First run downloads ~50MB and caches it under
    # ~/scikit_learn_data — later runs load from disk instantly.
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

    # Expected: X is (70000, 784) — 70,000 images, each a flattened 28*28=784
    # pixel vector. y is (70000,) — one label per image.
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Pixel values are floats but in the raw 0-255 range, not yet normalized
    # to 0-1. We'll do that normalization ourselves in a later lesson so it's
    # clear it's a deliberate step, not something sklearn did for us.
    print("X dtype:", X.dtype)
    print("X min/max:", X.min(), X.max())

    # Labels arrive as strings ('5', '0', '4', ...) because fetch_openml
    # doesn't know they're meant to be numeric. Convert to integers so we
    # can do arithmetic/comparisons on them later (e.g. y == 3).
    print("y dtype (before):", y.dtype)
    print("y[:10] (before):", y[:10])
    y = y.astype(np.uint8)
    print("y[:10] (after): ", y[:10])

    # Visualize the first 9 digits. Each row of X is flattened; reshape
    # back to (28, 28) to display it as an actual image.
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        image = X[i].reshape(28, 28)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"label: {y[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("digit-recognition/lessons/01_sample_digits.png")
    print("Saved sample grid to digit-recognition/lessons/01_sample_digits.png")

    # Split into train/test. We hold out a test set because the real goal is
    # generalization to unseen digits, not memorizing the training images.
    # If we trained and evaluated on the same data, a model that just
    # memorized every example would score 100% while being useless in
    # practice. 60,000/10,000 is the traditional MNIST split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    assert X_train.shape[0] + X_test.shape[0] == 70000


if __name__ == "__main__":
    main()
