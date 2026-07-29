"""
Lesson 6 — PyTorch tensors and autograd

Concepts: tensors, requires_grad, the computation graph, .backward(),
comparing autograd's gradient against the hand-derived formula from
Lesson 3 to prove they're the same thing.
"""

import torch
import numpy as np


def main():
    torch.manual_seed(42)

    # A tiny toy example first: z = w*x + b, p = sigmoid(z),
    # loss = binary cross entropy against target y=1.
    # requires_grad=True tells PyTorch "track operations on this tensor,
    # I'll want its gradient later."
    x = torch.tensor([2.0, -1.0, 0.5])
    w = torch.tensor([0.1, 0.2, -0.3], requires_grad=True)
    b = torch.tensor(0.05, requires_grad=True)
    y = torch.tensor(1.0)

    z = torch.dot(w, x) + b
    p = torch.sigmoid(z)
    loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))

    print("forward pass:")
    print("  z =", z.item())
    print("  p =", p.item())
    print("  loss =", loss.item())

    # This single call walks the graph (loss -> p -> z -> w, b) backward,
    # applying the chain rule at each step, and fills in w.grad / b.grad.
    loss.backward()
    print("\nautograd's gradients:")
    print("  w.grad =", w.grad)
    print("  b.grad =", b.grad)

    # Lesson 3's hand-derived formula for sigmoid + cross-entropy was:
    # dL/dw = (p - y) * x,  dL/db = (p - y).
    # Recompute that by hand with plain numbers and compare.
    p_value = p.item()
    manual_grad_w = (p_value - y.item()) * x.numpy()
    manual_grad_b = p_value - y.item()
    print("\nour hand-derived formula, same numbers:")
    print("  manual dL/dw =", manual_grad_w)
    print("  manual dL/db =", manual_grad_b)
    print("\n(these should match autograd's output above almost exactly)")

    # A second, more useful example: differentiate through TWO layers,
    # which is exactly what Lesson 4's manual backprop had to do by hand
    # with explicit chain-rule bookkeeping (dA1, dZ1, ...). Autograd
    # handles arbitrary depth the same way, automatically.
    print("\n--- two-layer example (mirrors Lesson 4's architecture) ---")
    X = torch.randn(4, 784)            # a fake mini-batch of 4 flattened images
    # Scale BEFORE calling requires_grad_(), not after: `torch.randn(...,
    # requires_grad=True) * 0.01` makes the *scaled* tensor a non-leaf
    # (the output of a multiply), and PyTorch only auto-populates .grad
    # on leaf tensors. Scaling first, then marking requires_grad, keeps
    # W1/W2 as leaves so their gradients actually get filled in.
    W1 = (torch.randn(784, 128) * 0.01).requires_grad_()
    b1 = torch.zeros(128, requires_grad=True)
    W2 = (torch.randn(128, 10) * 0.01).requires_grad_()
    b2 = torch.zeros(10, requires_grad=True)
    target = torch.tensor([3, 7, 1, 9])  # fake true labels for the 4 images

    A1 = torch.relu(X @ W1 + b1)
    logits = A1 @ W2 + b2
    loss2 = torch.nn.functional.cross_entropy(logits, target)
    loss2.backward()

    print("loss:", loss2.item())
    print("W1.grad shape:", W1.grad.shape, " (one gradient per weight, computed automatically)")
    print("W2.grad shape:", W2.grad.shape)
    print("\nNo manual dZ2 = A2 - y_onehot, no manual dA1 = dZ2 @ W2.T --")
    print("autograd derived all of it from the forward pass alone.")


if __name__ == "__main__":
    main()
