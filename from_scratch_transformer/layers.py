import torch
import numpy as np

class Layer:
    def __init__(self, rows, cols):
        # General superclass for all layers
        self.rows = rows
        self.cols = cols
        self.output = torch.zeros(rows, cols)
        self.grad = torch.zeros(rows, cols)

    def accumulate_grad(self, grad):
        # Accumulate gradient into this layer
        self.grad += grad

    def clear_grad(self):
        # Reset gradient to zero
        self.grad = torch.zeros_like(self.output)

    def step(self, learning_rate):
        # Default: no parameters to update
        pass

class Input(Layer):
    def __init__(self, rows, cols):
        super().__init__(rows, cols)

    def set(self, tensor):
        # Accept any 2D tensor [T, D]
        if tensor.ndim != 2:
            raise ValueError(f"Input.set expected 2D tensor, got {tensor.ndim}D")
        self.output = tensor
        self.grad = torch.zeros_like(tensor)
        self.rows, self.cols = tensor.shape

    def randomize(self):
        # Random normal init
        self.output = torch.randn(self.rows, self.cols) * 0.1
        self.grad = torch.zeros_like(self.output)

    def backward(self):
        # No-op: gradients accumulate externally
        pass

    def step(self, learning_rate):
        # SGD update for parameter inputs
        self.output -= learning_rate * self.grad

class Linear(Layer):
    """
    Linear layer Y = X @ W + b, for 2D X.
    """
    def __init__(self, X, W, b):
        super().__init__(0, 0)
        self.X = X
        self.W = W
        self.b = b
        self.output = None
        self.grad = None

    def forward(self):
        out = self.X.output @ self.W.output + self.b.output
        self.output = out
        self.grad = torch.zeros_like(out)
        return self

    def backward(self):
        # dX, dW, db
        dX = self.grad @ self.W.output.T
        dW = self.X.output.T @ self.grad
        db = self.grad.sum(dim=0, keepdim=True)
        self.X.accumulate_grad(dX)
        self.W.accumulate_grad(dW)
        self.b.accumulate_grad(db)

    def step(self, learning_rate):
        self.W.step(learning_rate)
        self.b.step(learning_rate)

    def clear_grad(self):
        self.W.clear_grad()
        self.b.clear_grad()

class ReLU(Layer):
    """
    ReLU activation for 2D tensors.
    """
    def __init__(self, prev_layer):
        super().__init__(0, 0)
        self.prev = prev_layer
        self.output = None
        self.grad = None

    def forward(self):
        out = torch.clamp(self.prev.output, min=0)
        self.output = out
        self.grad = torch.zeros_like(out)
        return self

    def backward(self):
        grad_in = self.grad * (self.prev.output > 0).float()
        self.prev.accumulate_grad(grad_in)

class Sum(Layer):
    """
    Elementwise sum of two 2D layers.
    """
    def __init__(self, a, b):
        if a.output.shape != b.output.shape:
            raise ValueError(f"Sum operands must have same shape, got {a.output.shape} vs {b.output.shape}")
        rows, cols = a.output.shape
        super().__init__(rows, cols)
        self.a = a
        self.b = b

    def forward(self):
        out = self.a.output + self.b.output
        self.output = out
        self.grad = torch.zeros_like(out)
        return self

    def backward(self):
        self.a.accumulate_grad(self.grad)
        self.b.accumulate_grad(self.grad)

class Concat(Layer):
    """
    Concatenate list of 2D layers along the feature (cols) dimension.
    """
    def __init__(self, layers_list):
        shapes = [l.output.shape for l in layers_list]
        # rows must match
        rows = shapes[0][0]
        for s in shapes:
            if s[0] != rows:
                raise ValueError("All layers must have same number of rows to concat")
        cols = sum(s[1] for s in shapes)
        super().__init__(rows, cols)
        self.layers_list = layers_list

    def forward(self):
        outs = [l.output for l in self.layers_list]
        self.output = torch.cat(outs, dim=1)
        self.grad = torch.zeros_like(self.output)
        return self

    def backward(self):
        sizes = [l.output.shape[1] for l in self.layers_list]
        splits = torch.split(self.grad, sizes, dim=1)
        for l, g in zip(self.layers_list, splits):
            l.accumulate_grad(g)

class Softmax(Layer):
    """
    Combined softmax activation and cross-entropy loss.
    Assumes inputs are 2D [T, V].
    """
    def __init__(self, x_layer, y_layer):
        # x_layer.output: logits [T, V]; y_layer.output: one-hot [T, V]
        T, V = x_layer.output.shape
        super().__init__(T, V)
        self.x = x_layer
        self.y = y_layer
        self.classifications = None

    def forward(self):
        logits = self.x.output
        # stable softmax
        shift = logits - logits.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(shift)
        probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
        self.classifications = probs
        # cross-entropy
        loss = -(self.y.output * torch.log(probs + 1e-9)).sum(dim=1).mean()
        self.output = loss
        self.grad = torch.tensor(1.0)  # dLoss/dLoss
        return self

    def backward(self):
        # dJ/dz = p - t
        grad_logits = (self.classifications - self.y.output) / self.y.rows
        self.x.accumulate_grad(grad_logits)

