"""
Lesson 5 — Evaluate and predict

Concepts: accuracy vs. confusion matrix, finding the model's systematic
mistakes, predicting on individual images.

Loads the weights Lesson 4 trained (no retraining) and digs into *what
kind* of mistakes the network makes, not just how many.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def predict(X, W1, b1, W2, b2):
    A1 = relu(X @ W1 + b1)
    A2 = softmax(A1 @ W2 + b2)
    return A2.argmax(axis=1)


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(np.uint8)
    X = X / 255.0

    _, X_test, _, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )

    weights = np.load("digit-recognition/lessons/04_trained_weights.npz")
    W1, b1, W2, b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]

    y_pred = predict(X_test, W1, b1, W2, b2)
    accuracy = (y_pred == y_test).mean()
    print(f"Overall test accuracy: {accuracy:.4f}\n")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true digit, cols=predicted digit):")
    print(cm)

    # Per-digit accuracy: is the model equally good at every digit, or
    # does it struggle more with some than others?
    print("\nPer-digit accuracy:")
    for digit in range(10):
        digit_total = cm[digit].sum()
        digit_correct = cm[digit, digit]
        print(f"  {digit}: {digit_correct}/{digit_total} = {digit_correct/digit_total:.4f}")

    # Find the most-confused off-diagonal pairs -- the specific mistakes
    # the model makes most often.
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    print("\nTop 5 most common mistakes (true -> predicted, count):")
    flat_indices = np.argsort(off_diagonal, axis=None)[::-1][:5]
    for idx in flat_indices:
        true_digit, pred_digit = np.unravel_index(idx, cm.shape)
        count = off_diagonal[true_digit, pred_digit]
        print(f"  {true_digit} -> {pred_digit}: {count} times")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="count")
    plt.xlabel("predicted digit")
    plt.ylabel("true digit")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title("Confusion matrix")
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig("digit-recognition/lessons/05_confusion_matrix.png")
    print("\nSaved confusion matrix plot to digit-recognition/lessons/05_confusion_matrix.png")

    # Visualize actual misclassified examples: true label vs what the
    # network guessed, so you can see *why* it was a reasonable mistake.
    misclassified = np.where(y_pred != y_test)[0]
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        idx = misclassified[i]
        image = X_test[idx].reshape(28, 28)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"true: {y_test[idx]}  pred: {y_pred[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("digit-recognition/lessons/05_misclassified.png")
    print("Saved misclassified examples to digit-recognition/lessons/05_misclassified.png")


if __name__ == "__main__":
    main()
