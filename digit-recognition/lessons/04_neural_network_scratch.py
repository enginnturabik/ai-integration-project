"""
Lesson 4 — A multi-layer neural network, from scratch

Concepts: hidden layers, ReLU, softmax, one-hot labels, categorical
cross-entropy, backpropagation, mini-batch gradient descent.

Architecture: 784 inputs -> 128 hidden neurons (ReLU) -> 10 outputs (softmax).
This is the full 10-digit classifier, no shortcuts, pure NumPy.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    # Subtract the row max before exponentiating purely for numerical
    # stability (keeps exponents from overflowing) -- doesn't change the
    # result since softmax is shift-invariant.
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(y, n_classes=10):
    encoded = np.zeros((y.shape[0], n_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded


def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))


class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, rng):
        # Small random weights break symmetry: if all weights started
        # identical, every hidden neuron would compute the same thing and
        # get the same gradient forever, making extra neurons pointless.
        # Scaling by sqrt(1/n_input) (a simplified "He initialization")
        # keeps the initial activations from exploding or vanishing.
        self.W1 = rng.normal(0, np.sqrt(1 / n_input), size=(n_input, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), size=(n_hidden, n_output))
        self.b2 = np.zeros(n_output)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_onehot, learning_rate):
        n = X.shape[0]

        # Output layer error. For softmax + cross-entropy together, the
        # gradient of the loss w.r.t. Z2 simplifies to exactly this --
        # same clean "prediction minus truth" pattern as logistic regression.
        dZ2 = self.A2 - y_onehot                    # (n, 10)
        dW2 = self.A1.T @ dZ2 / n                    # (128, 10)
        db2 = dZ2.mean(axis=0)                       # (10,)

        # Push the error backward through W2 to find the hidden layer's
        # share of the blame, then apply the chain rule through ReLU.
        dA1 = dZ2 @ self.W2.T                         # (n, 128)
        dZ1 = dA1 * relu_derivative(self.Z1)          # (n, 128)
        dW1 = X.T @ dZ1 / n                           # (784, 128)
        db1 = dZ1.mean(axis=0)                        # (128,)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def main():
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(np.uint8)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )
    y_train_onehot = one_hot(y_train)

    rng = np.random.default_rng(42)
    net = NeuralNetwork(n_input=784, n_hidden=128, n_output=10, rng=rng)

    n_epochs = 20
    batch_size = 128
    learning_rate = 0.5
    n_samples = X_train.shape[0]

    train_loss_history = []
    test_accuracy_history = []

    for epoch in range(n_epochs):
        # Shuffle each epoch so mini-batches aren't the same fixed groups
        # every time -- otherwise the network could latch onto batch order.
        perm = rng.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train_onehot[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            net.forward(X_batch)
            net.backward(X_batch, y_batch, learning_rate)

        # Measure progress once per epoch on the full training set (loss)
        # and test set (accuracy), not per mini-batch -- cheaper and less
        # noisy than tracking every single step.
        train_pred = net.forward(X_train)
        train_loss = categorical_cross_entropy(y_train_onehot, train_pred)
        test_accuracy = (net.predict(X_test) == y_test).mean()

        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_accuracy)
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  test accuracy {test_accuracy:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_loss_history)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.set_title("Training loss")
    ax2.plot(test_accuracy_history)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    ax2.set_title("Test accuracy")
    plt.tight_layout()
    plt.savefig("digit-recognition/lessons/04_training_curves.png")
    print("Saved training curves to digit-recognition/lessons/04_training_curves.png")

    # Save the trained weights so Lesson 5 can load this exact model
    # without retraining from scratch.
    np.savez(
        "digit-recognition/lessons/04_trained_weights.npz",
        W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
    )
    print("Saved trained weights to digit-recognition/lessons/04_trained_weights.npz")


if __name__ == "__main__":
    main()
