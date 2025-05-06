# from_scratch_transformer/rms_norm.py
from . import layers
import torch

class RMSNorm(layers.Layer):

    """
    RMSNorm layer for Transformer.
    This class normalizes the input using the root mean square of the input values.
    It is similar to LayerNorm but uses RMS instead of mean and variance.
    """
    def __init__(self, x_layer, gamma):
        super().__init__(0,0)
        self.x = x_layer
        self.gamma = gamma

    def forward(self):
        x = self.x.output 
        T, D = x.shape
        std = torch.sqrt((x**2).sum(dim=1, keepdim=True)/D)
        out = x/(std+1e-9) * self.gamma.output
        self.output = out
        self.grad = torch.zeros_like(out)
        return self

    def backward(self):
        x = self.x.output; g = self.gamma.output; grad = self.grad
        dJdg = ((x/(torch.sqrt((x**2).sum(dim=1, keepdim=True)/x.shape[1])+1e-9))*grad).sum(dim=0, keepdim=True)
        simple = grad * g / torch.sqrt((x**2).sum(dim=1, keepdim=True)/x.shape[1])
        dJdx = simple - x*(simple*x).sum(dim=1,keepdim=True)/(x.shape[1]* (torch.sqrt((x**2).sum(dim=1, keepdim=True)/x.shape[1])+1e-9)**2)
        self.gamma.accumulate_grad(dJdg)
        self.x.accumulate_grad(dJdx)

    def step(self, learning_rate):
        self.gamma.step(learning_rate)

    def clear_grad(self):
        self.gamma.clear_grad()