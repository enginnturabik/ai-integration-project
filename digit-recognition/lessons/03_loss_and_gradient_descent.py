"""
Lesson 3 — Loss and gradient descent

Concepts: loss functions, gradients, learning rate, how a model "learns".

Lesson 2 used sklearn to find good weights for a single neuron. Here we
implement the training loop ourselves: binary cross-entropy loss and
gradient descent, from scratch in NumPy, on the same "is this a 5?" task.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    # Clip predictions away from exactly 0 or 1 so log() never hits -inf.
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(np.uint8)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )
    y_train_is5 = (y_train == 5).astype(np.float64)
    y_test_is5 = (y_test == 5).astype(np.uint8)

    n_features = X_train.shape[1]
    rng = np.random.default_rng(42)
    # Small random weights, not zeros: with all-zero weights every neuron
    # would start identical, which is harmless here (single neuron) but is
    # the reason you'll never zero-initialize a multi-neuron network in
    # Lesson 4 (symmetry: every neuron would learn the same thing).
    w = rng.normal(0, 0.01, size=n_features)
    b = 0.0

    learning_rate = 0.5
    n_epochs = 500
    n_samples = X_train.shape[0]
    loss_history = []

    for epoch in range(n_epochs):
        # Forward pass: compute the neuron's current predictions for the
        # whole training set at once (vectorized, no Python loop over rows).
        z = X_train @ w + b
        p = sigmoid(z)

        loss = binary_cross_entropy(y_train_is5, p)
        loss_history.append(loss)

        # Gradient of cross-entropy loss w.r.t. w and b, averaged over all
        # training examples. This is the "(prediction - truth) * input"
        # rule derived in the lesson notes above.
        error = p - y_train_is5  # shape (n_samples,)
        grad_w = (X_train.T @ error) / n_samples
        grad_b = np.mean(error)

        # Gradient descent step: move w and b in the direction that
        # decreases the loss.
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss:.4f}")

    # Plot the loss curve. It should fall sharply at first, then flatten
    # out as the neuron approaches its best achievable fit.
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy loss")
    plt.title("Training loss over time")
    plt.tight_layout()
    plt.savefig("digit-recognition/lessons/03_loss_curve.png")
    print("Saved loss curve to digit-recognition/lessons/03_loss_curve.png")

    # Evaluate on the held-out test set using our from-scratch weights.
    z_test = X_test @ w + b
    p_test = sigmoid(z_test)
    pred_test = (p_test >= 0.5).astype(np.uint8)
    accuracy = (pred_test == y_test_is5).mean()
    print(f"\nTest accuracy with from-scratch gradient descent: {accuracy:.4f}")
    print("(Lesson 2's sklearn version got 0.9737 — we should be in the same ballpark.)")


if __name__ == "__main__":
    main()
