# Handwritten Digit Recognition — From Zero

A hands-on curriculum for building a digit classifier twice: first with nothing but
NumPy (so you see exactly what a neural network is), then with PyTorch (so you see
what the framework is automating for you).

Dataset: MNIST — 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.
It's the "hello world" of image classification.

## How this works

Each lesson is a script in `lessons/` with a docstring explaining the concept, TODOs
marking what you write, and hints. Write the code yourself, run it, and bring questions
or your solution back for review before moving to the next lesson. Nothing here is
solved for you in advance.

## Roadmap

### Phase 1 — NumPy from scratch (understand the math)

| Lesson | Concept |
|---|---|
| `01_load_and_explore_data.py` | What MNIST is: features, labels, train/test split, pixel arrays |
| `02_single_neuron.py` | A single neuron: weights, bias, dot product, sigmoid, binary classification |
| `03_loss_and_gradient_descent.py` | Loss functions, gradients, learning rate — how a model "learns" |
| `04_neural_network_scratch.py` | Multi-layer network: forward pass, backpropagation, training loop |
| `05_evaluate_and_predict.py` | Accuracy, confusion matrix, predicting on new/custom digits |

### Phase 2 — PyTorch (the industry-standard tool)

| Lesson | Concept |
|---|---|
| `06_pytorch_tensors_autograd.py` | Tensors, autograd (automatic differentiation) |
| `07_pytorch_mlp.py` | Rebuilding lesson 4's network in PyTorch — same model, ~10x less code |
| `08_pytorch_cnn.py` | Convolutional layers — why CNNs beat plain MLPs on images |
| `09_predict_your_own_handwriting.py` | Run the trained model on a digit you draw or photograph |

## Concept glossary

- **Feature / label** — the input (pixels, `X`) vs. the answer we want predicted (digit, `y`).
- **Train/test split** — holding out data the model never trains on, so accuracy measures generalization, not memorization.
- **Neuron** — computes `z = w·x + b` (weighted sum + bias), then applies an activation function.
- **Sigmoid** — `σ(z) = 1/(1+e⁻ᶻ)`, squashes any real number into `(0,1)`, read as a probability. Used for binary (yes/no) output.
- **Softmax** — sigmoid generalized to multiple classes; turns N raw scores into a probability distribution that sums to 1.
- **ReLU** — `max(0, z)`, the standard hidden-layer activation; avoids the vanishing-gradient problems sigmoid has once layers stack up.
- **Loss function** — one number measuring how wrong the model's predictions are right now. Binary cross-entropy for yes/no, categorical cross-entropy for multi-class.
- **Gradient descent** — repeatedly nudging weights in the direction that decreases the loss, scaled by a learning rate.
- **Backpropagation** — the chain rule applied layer by layer, back-to-front, to get the gradient for every weight in a multi-layer network.
- **Mini-batch** — training on small random chunks of data per step instead of the whole dataset at once; makes large datasets tractable.
- **Overfitting** — training loss keeps falling while test accuracy plateaus or worsens; the model is memorizing rather than generalizing.
- **Confusion matrix** — a grid of true label vs. predicted label counts; reveals *which* mistakes a model makes, not just how many.
- **Tensor / autograd** — PyTorch's array type that can track operations performed on it and compute gradients automatically via `.backward()`, replacing hand-derived backprop.
- **Convolution / weight sharing** — a small filter slid across every position of an image, reusing the same weights everywhere; preserves spatial structure that flattening destroys, with far fewer parameters than a fully-connected layer.
- **Pooling** — downsampling (e.g. max over 2x2 blocks) that shrinks the spatial size and adds a bit of translation invariance.

## Results achieved

| Lesson | Model | Test accuracy |
|---|---|---|
| 2 | Single neuron (sklearn), binary "is this a 5?" | 97.4% |
| 3 | Single neuron, hand-written gradient descent | 96.8% |
| 4 | 2-layer MLP (784→128→10), backprop from scratch in NumPy | 97.8% |
| 7 | Same MLP architecture, rebuilt in PyTorch | 97.8% |
| 8 | CNN (2 conv+pool blocks + MLP head), PyTorch | 99.0% |

## Setup

```bash
venv\Scripts\activate
pip install -r digit-recognition/requirements.txt
```
