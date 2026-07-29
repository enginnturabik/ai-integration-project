"""
Lesson 8 — Convolutional Neural Network (CNN)

Concepts: why flattening images (what Lessons 4 and 7 did) throws away
spatial structure, convolutional filters, weight sharing, pooling.

The MLP in Lesson 7 flattens each 28x28 image into a 784-length vector.
That destroys locality: pixel (5,5) and its neighbor (5,6) end up in
unrelated positions of the vector, so the network has to independently
relearn "an edge looks like this" at every possible position in the
image. A convolutional layer instead slides one small filter (e.g. 3x3
weights) across every position of the image, reusing the SAME weights
everywhere. That means: (1) far fewer parameters than a fully-connected
layer, and (2) a pattern learned in one part of the image is
automatically recognized anywhere else too (translation invariance).

Architecture:
  Conv(1->16, 3x3) -> ReLU -> MaxPool(2x2)   [28x28 -> 14x14]
  Conv(16->32, 3x3) -> ReLU -> MaxPool(2x2)  [14x14 -> 7x7]
  Flatten -> Linear(32*7*7 -> 128) -> ReLU -> Linear(128 -> 10)
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=1 with a 3x3 kernel keeps the spatial size unchanged
        # (28x28 in, 28x28 out) so only the MaxPool layers shrink it.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # halves height and width
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (N,1,28,28) -> (N,16,14,14)
        x = self.pool(torch.relu(self.conv2(x)))  # -> (N,32,7,7)
        x = x.flatten(start_dim=1)                 # -> (N, 32*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

    # Conv2d expects (N, channels, height, width), not a flat 784 vector --
    # reshape back to the 2D image the convolution needs to slide across.
    X_train_img = X_train.reshape(-1, 1, 28, 28)
    X_test_img = X_test.reshape(-1, 1, 28, 28)

    train_ds = TensorDataset(torch.from_numpy(X_train_img), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    X_test_t = torch.from_numpy(X_test_img).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_accuracy = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()
        print(f"epoch {epoch:2d}  test accuracy {test_accuracy:.4f}")

    print("\nLesson 4 (NumPy MLP):    0.9778")
    print("Lesson 7 (PyTorch MLP):  see Lesson 7 output")
    print("Lesson 8 (PyTorch CNN):  see above -- should be noticeably higher")

    torch.save(model.state_dict(), "digit-recognition/lessons/08_cnn_weights.pt")
    print("Saved model weights to digit-recognition/lessons/08_cnn_weights.pt")


if __name__ == "__main__":
    main()
