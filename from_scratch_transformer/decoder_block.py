# from_scratch_transformer/decoder_block.py
from . import layers
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .rms_norm import RMSNorm

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        # gamma parameters for RMSNorm
        self.gamma1 = layers.Input(rows=1, cols=d_model); self.gamma1.randomize()
        self.norm1 = RMSNorm(x_layer=layers.Input(rows=1, cols=d_model), gamma=self.gamma1)
        self.gamma2 = layers.Input(rows=1, cols=d_model); self.gamma2.randomize()
        self.norm2 = RMSNorm(x_layer=layers.Input(rows=1, cols=d_model), gamma=self.gamma2)

    def forward(self, x_layer, mask=None):
        attn_out_layer, _ = self.self_attn.forward(x_layer, mask)
        sum1 = layers.Sum(x_layer, attn_out_layer)
        sum1.forward()
        self.norm1.x = sum1; self.norm1.forward()
        ff_out_layer = self.ff.forward(self.norm1)
        sum2 = layers.Sum(self.norm1, ff_out_layer)
        sum2.forward()
        self.norm2.x = sum2; self.norm2.forward()
        return self.norm2

    def backward(self):
        self.norm2.backward()
        self.ff.backward()
        self.norm1.backward()
        self.self_attn.backward()

    def step(self, learning_rate):
        self.gamma1.step(learning_rate= learning_rate)
        self.gamma2.step(learning_rate= learning_rate)
        self.self_attn.step(learning_rate= learning_rate)
        self.ff.step(learning_rate= learning_rate)


    def clear_grad(self):
        self.gamma1.clear_grad()
        self.gamma2.clear_grad()
        self.self_attn.clear_grad()
        self.ff.clear_grad()
        self.norm1.clear_grad()
        self.norm2.clear_grad()


