"""
California: is this district expensive? (above median house price)
Inputs: longitude, latitude only.

A neural network built from scratch with numpy:
  2 inputs -> hidden layers (ReLU) -> 1 output (sigmoid)
  loss: binary cross-entropy
  training: mini-batches + Adam optimizer
"""
import numpy as np
# same data, same 80/20 split as the neuron
from neuron_california import import_data

LAYERS = [2, 128, 128, 128, 1]   # input, hidden..., output
LEARNING_RATE = 0.003
BATCH_SIZE = 64
EPOCHS = 150
SEED = 42


# ---------- data ----------
def load():
    train, test = import_data()
    X_train = np.array([x for x, _ in train])
    y_train = np.array([y for _, y in train])
    X_test = np.array([x for x, _ in test])
    y_test = np.array([y for _, y in test])

    # centre and scale using TRAIN statistics only
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, y_train, X_test, y_test


# ---------- network ----------
def init_network(layers, rng):
    """Random weights (He initialisation, suits ReLU), zero biases."""
    weights, biases = [], []
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        weights.append(rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)))
        biases.append(np.zeros(n_out))
    return weights, biases


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(weights, biases, X):
    """Returns every layer's output; the last one is P(expensive)."""
    activations = [X]
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ W + b
        is_last = i == len(weights) - 1
        activations.append(sigmoid(z) if is_last else relu(z))
    return activations


def backward(weights, activations, y):
    """Backpropagation. Same chain rule as the XOR network, with ReLU in the hidden layers."""
    n = len(y)
    grads_W, grads_b = [None] * len(weights), [None] * len(weights)

    delta = (activations[-1].ravel() - y).reshape(-1, 1) / \
        n     # output: p - y (BCE + sigmoid)
    for L in range(len(weights) - 1, -1, -1):
        grads_W[L] = activations[L].T @ delta
        grads_b[L] = delta.sum(axis=0)
        if L > 0:
            # ReLU derivative: 1 if z > 0
            delta = (delta @ weights[L].T) * (activations[L] > 0)
    return grads_W, grads_b


def bce_loss(p, y):
    eps = 1e-12
    p = p.ravel()
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


# ---------- Adam optimizer ----------
class Adam:
    """Gradient descent where each weight gets its own step size, based on its recent gradients."""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------- train / evaluate ----------
def evaluate(weights, biases, X, y):
    p = forward(weights, biases, X)[-1].ravel()
    return ((p >= 0.5) == y).mean()


def train(X, y, X_test, y_test):
    rng = np.random.default_rng(SEED)
    weights, biases = init_network(LAYERS, rng)
    optimizer = Adam(weights + biases, LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y))                 # shuffle every epoch
        for start in range(0, len(y), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            activations = forward(weights, biases, X[batch])
            gW, gb = backward(weights, activations, y[batch])
            optimizer.step(weights + biases, gW + gb)

        if epoch % 10 == 0 or epoch == 1:
            loss = bce_loss(forward(weights, biases, X)[-1], y)
            print(f"Epoch {epoch:3d} | loss {loss:.4f} | "
                  f"train {evaluate(weights, biases, X, y) * 100:.1f}% | "
                  f"test {evaluate(weights, biases, X_test, y_test) * 100:.1f}%")
    return weights, biases


def main():
    X_train, y_train, X_test, y_test = load()
    print(f"network: {LAYERS}")
    weights, biases = train(X_train, y_train, X_test, y_test)
    print(
        f"\nTrain accuracy: {evaluate(weights, biases, X_train, y_train) * 100:.1f}%")
    print(
        f"Test accuracy : {evaluate(weights, biases, X_test, y_test) * 100:.1f}%")


if __name__ == "__main__":
    main()
