"""
Lesson 9 — Predict on your own handwriting

Concepts: preprocessing new, real-world input into the exact shape and
distribution a trained model expects. A model trained on 28x28 grayscale
images normalized to [0, 1], white-digit-on-black, will fail on anything
that doesn't match that format -- this lesson makes that pipeline explicit.

Draw a digit with your mouse in the window that appears, then click
Predict. Uses the CNN trained in Lesson 8.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

CANVAS_SIZE = 280   # 10x the model's native 28x28, just for a comfortable drawing area
MODEL_SIZE = 28
BRUSH_RADIUS = 10


class CNN(nn.Module):
    """Must match Lesson 8's architecture exactly -- we're loading its
    trained weights into this, and shapes have to line up layer for layer."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DigitApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Draw a digit (0-9)")

        # Real drawing surface: a PIL image kept in sync with the on-screen
        # canvas, so we can hand it to the model directly -- no screen
        # capture needed. Black background, white strokes, matching
        # MNIST's white-digit-on-black convention exactly.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.result_label = ttk.Label(root, text="Draw a digit, then click Predict", font=("Segoe UI", 14))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=8)

        ttk.Button(root, text="Predict", command=self.predict).grid(row=2, column=0, sticky="ew")
        ttk.Button(root, text="Clear", command=self.clear).grid(row=2, column=1, sticky="ew")

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit, then click Predict")

    def preprocess(self):
        # Downscale from the 280x280 drawing surface to the 28x28 the
        # model was trained on. LANCZOS gives smoother downsampling than
        # nearest-neighbor, closer to how real MNIST digits look after
        # having been scanned and downsampled themselves.
        small = self.image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
        array = np.array(small, dtype=np.float32) / 255.0  # match training normalization
        tensor = torch.from_numpy(array).reshape(1, 1, MODEL_SIZE, MODEL_SIZE)
        return tensor

    def predict(self):
        x = self.preprocess()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        predicted = probs.argmax().item()
        confidence = probs[predicted].item()
        self.result_label.config(text=f"Prediction: {predicted}  (confidence: {confidence:.1%})")
        print("Full probability distribution:")
        for digit, p in enumerate(probs.tolist()):
            print(f"  {digit}: {p:.4f}")


def main():
    model = CNN()
    model.load_state_dict(torch.load("digit-recognition/lessons/08_cnn_weights.pt", map_location="cpu"))

    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
