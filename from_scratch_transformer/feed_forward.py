# from_scratch_transformer/feed_forward.py
from .layers import Linear, ReLU, Input

class FeedForward:
    """
    Custom feed-forward: Linear -> ReLU -> Linear, no batch dimension (2D only).
    """
    def __init__(self, d_model, d_ff):
        # Initialize weight and bias parameters as Input layers
        self.W1 = Input(d_model, d_ff)
        self.W1.randomize()
        self.b1 = Input(1, d_ff)
        self.b1.randomize()

        self.W2 = Input(d_ff, d_model)
        self.W2.randomize()
        self.b2 = Input(1, d_model)
        self.b2.randomize()

        # Linear/ReLU layers
        self.lin1 = None
        self.relu_layer = None
        self.lin2 = None

    def forward(self, x_layer):
        # First linear
        self.lin1 = Linear(X=x_layer, W=self.W1, b=self.b1)
        self.lin1.forward()
        # ReLU activation
        self.relu_layer = ReLU(prev_layer=self.lin1)
        self.relu_layer.forward()
        # Second linear
        self.lin2 = Linear(X=self.relu_layer, W=self.W2, b=self.b2)
        self.lin2.forward()
        return self.lin2

    def backward(self):
        # Backprop
        self.lin2.backward()
        self.relu_layer.backward()
        self.lin1.backward()

    def step(self, learning_rate):
        # Update parameters
        self.W1.step(learning_rate= learning_rate)
        self.b1.step(learning_rate= learning_rate)
        self.W2.step(learning_rate= learning_rate)
        self.b2.step(learning_rate= learning_rate)

    def clear_grad(self):
        # Clear gradients for all components
        self.W1.clear_grad()
        self.b1.clear_grad()
        self.W2.clear_grad()
        self.b2.clear_grad()
        if self.lin1:
            self.lin1.clear_grad()
        if self.relu_layer:
            self.relu_layer.clear_grad()
        if self.lin2:
            self.lin2.clear_grad()
