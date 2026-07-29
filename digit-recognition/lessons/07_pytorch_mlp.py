"""
Lesson 7 — Rebuilding Lesson 4's network in PyTorch

Concepts: nn.Module, nn.Linear, nn.CrossEntropyLoss, optimizers,
DataLoader/mini-batching -- the same 784->128->10 network as Lesson 4,
but with the framework handling initialization, forward-pass bookkeeping,
backward pass, and parameter updates.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    """Same architecture as Lesson 4: 784 -> 128 (ReLU) -> 10.

    nn.Linear(in, out) IS "a layer of neurons": it owns a (in, out) weight
    matrix and an (out,) bias, initialized sensibly for you -- exactly the
    W1/b1/W2/b2 we allocated and initialized by hand in Lesson 4.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)  # raw logits -- no softmax here, see note below
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(np.int64)
    X = (X / 255.0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = MLP().to(device)

    # nn.CrossEntropyLoss expects raw logits (not softmax probabilities)
    # and integer labels (not one-hot) -- it applies log-softmax
    # internally, combined in a numerically stabler way than doing it in
    # two separate steps. That's why forward() above returns logits.
    criterion = nn.CrossEntropyLoss()

    # The optimizer owns the parameter-update rule. SGD here does exactly
    # our manual "w -= learning_rate * grad_w" from Lessons 3-4; swapping
    # in torch.optim.Adam would use a fancier per-parameter adaptive rule,
    # same interface.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()          # clear gradients from last step
            logits = model(X_batch)        # forward pass
            loss = criterion(logits, y_batch)
            loss.backward()                # autograd computes all gradients
            optimizer.step()                # apply the update rule

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_accuracy = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()
        print(f"epoch {epoch:2d}  test accuracy {test_accuracy:.4f}")

    print("\nLesson 4 (pure NumPy) got 0.9778 test accuracy with the same architecture.")
    torch.save(model.state_dict(), "digit-recognition/lessons/07_mlp_weights.pt")
    print("Saved model weights to digit-recognition/lessons/07_mlp_weights.pt")


if __name__ == "__main__":
    main()
