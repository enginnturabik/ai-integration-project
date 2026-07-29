"""
Lesson 2 — A single neuron

Concepts: weights, bias, dot product, sigmoid activation, binary
classification.

A neuron computes z = w . x + b, then squashes z through an activation
function. Here we use sigmoid, which maps any real number to (0, 1) so
the output can be read as "probability the answer is yes".

One neuron only answers yes/no questions, so we frame digit recognition
as a binary problem: "is this image a 5, or not?" We let sklearn's
LogisticRegression (which IS a single neuron) learn the weights, then
manually recompute its forward pass by hand to show exactly what's
happening inside: dot product, add bias, apply sigmoid.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(np.uint8)

    # Normalize pixels to [0, 1]. Raw values are 0-255; large inputs make
    # the dot product z = w.x + b swing to extreme values, which saturates
    # sigmoid (its output barely changes for very large |z|) and makes
    # training unstable. Scaling to [0, 1] keeps z in a reasonable range.
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )

    # Binary framing: 1 if the digit is a 5, 0 otherwise.
    y_train_is5 = (y_train == 5).astype(np.uint8)
    y_test_is5 = (y_test == 5).astype(np.uint8)

    # Let sklearn's single-neuron model learn the weights (training is
    # Lesson 3's topic — for now we just want a trained w and b to inspect).
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train_is5)

    w = clf.coef_[0]      # shape (784,) — one weight per pixel
    b = clf.intercept_[0]  # a single scalar bias
    print("w shape:", w.shape, " b:", b)

    # This is the entire neuron, written out by hand: dot product, add
    # bias, squash with sigmoid. No sklearn involved in this line.
    z = X_test @ w + b
    our_probability = sigmoid(z)
    our_prediction = (our_probability >= 0.5).astype(np.uint8)

    # Compare against sklearn's own predict_proba/predict to prove our
    # hand-rolled forward pass computes the exact same thing.
    sklearn_probability = clf.predict_proba(X_test)[:, 1]
    sklearn_prediction = clf.predict(X_test)

    max_prob_diff = np.abs(our_probability - sklearn_probability).max()
    print("Max difference between our probabilities and sklearn's:", max_prob_diff)
    print("Our predictions match sklearn's:", np.array_equal(our_prediction, sklearn_prediction))

    accuracy = (our_prediction == y_test_is5).mean()
    print(f"Accuracy on 'is this a 5?': {accuracy:.4f}")

    # Sanity check: a neuron that always guesses "not 5" would still score
    # ~90% accuracy, since only ~10% of digits are 5s. Compare against that
    # baseline so the number above means something.
    baseline_accuracy = (y_test_is5 == 0).mean()
    print(f"Baseline accuracy (always guess 'not 5'): {baseline_accuracy:.4f}")

    # Look at which pixels the neuron weighted most heavily. Reshaping w
    # back to (28, 28) shows *where* on the image it's looking for evidence
    # of a 5 (positive weight) vs evidence against (negative weight).
    print("\nStrongest positive weight (most '5-like' pixel):", w.max())
    print("Strongest negative weight (most 'not-5-like' pixel):", w.min())


if __name__ == "__main__":
    main()
