# from_scratch_transformer/positional_encoding.py
import torch
from . import layers
import math

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe  # [max_len, d_model]
        self.output = None
        self.grad = None
        self.x = None

    def forward(self, x_layer):
        # x_layer.output: [T, D]
        self.x = x_layer
        T, D = x_layer.output.shape
        out = x_layer.output + self.pe[:T]
        self.output = out
        self.grad = torch.zeros_like(out)
        return self

    def backward(self):
        self.x.accumulate_grad(self.grad)