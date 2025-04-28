# from_scratch_transformer/multihead_attention.py
import torch

from .layers import Linear, Input, Concat
import math

class MultiHeadAttention:
    """
    Multi-head self-attention without batch dimension (2D inputs: [T, D]).
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Learnable parameters as Input layers
        self.W_q = Input(d_model, d_model); self.W_q.randomize()
        self.b_q = Input(1,       d_model); self.b_q.randomize()
        self.W_k = Input(d_model, d_model); self.W_k.randomize()
        self.b_k = Input(1,       d_model); self.b_k.randomize()
        self.W_v = Input(d_model, d_model); self.W_v.randomize()
        self.b_v = Input(1,       d_model); self.b_v.randomize()
        self.W_o = Input(d_model, d_model); self.W_o.randomize()
        self.b_o = Input(1,       d_model); self.b_o.randomize()

        # Linear wrappers
        self.linear_q   = Linear(X=None, W=self.W_q, b=self.b_q)
        self.linear_k   = Linear(X=None, W=self.W_k, b=self.b_k)
        self.linear_v   = Linear(X=None, W=self.W_v, b=self.b_v)
        self.linear_out = Linear(X=None, W=self.W_o, b=self.b_o)

        # Placeholders for forward state
        self.qh = None
        self.kh = None
        self.vh = None
        self.attn = None
        self.mask = None
        self.concat = None

    def split_heads(self, x):
        # x: [T, D] -> [num_heads, T, d_k]
        T, D = x.shape
        return x.view(T, self.num_heads, self.d_k).transpose(0, 1)

    def combine_heads(self, x):
        # x: [num_heads, T, d_k] -> [T, D]
        T = x.shape[1]
        return x.transpose(0, 1).contiguous().view(T, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q,k,v: [num_heads, T, d_k]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v
        return output, attn

    def forward(self, x_layer, mask=None):
        # x_layer.output: [T, D]
        # 1) Linear projections
        self.linear_q.X = x_layer; self.linear_q.forward(); q = self.linear_q.output
        self.linear_k.X = x_layer; self.linear_k.forward(); k = self.linear_k.output
        self.linear_v.X = x_layer; self.linear_v.forward(); v = self.linear_v.output

        # 2) Split into heads
        self.qh = self.split_heads(q)
        self.kh = self.split_heads(k)
        self.vh = self.split_heads(v)
        self.mask = mask

        # 3) Scaled dot-product attention
        attn_out, self.attn = self.scaled_dot_product_attention(
            self.qh, self.kh, self.vh, mask
        )  # attn_out: [num_heads, T, d_k]

        # 4) Wrap each head output in an Input layer
        head_layers = []
        for h in range(self.num_heads):
            t = attn_out[h]  # [T, d_k]
            layer = Input(t.shape[0], t.shape[1])
            layer.set(t)
            head_layers.append(layer)

        # 5) Concatenate heads
        self.concat = Concat(head_layers)  # axis=1 (feature dim)
        self.concat.forward()  # [T, D]

        # 6) Final linear projection
        self.linear_out.X = self.concat
        self.linear_out.forward()  # [T, d_model]
        return self.linear_out, self.attn

    def backward(self):
        # Full backpropagation
        # 1) Final linear proj
        self.linear_out.backward()
        # 2) Concat split
        self.concat.backward()
        # 3) Gather head grads and stack
        head_grads = [hl.grad for hl in self.concat.layers_list]  # [T, d_k] each
        stacked    = torch.stack(head_grads, dim=0)               # [num_heads, T, d_k]

        # 4) d(out)=d(attn* v)
        grad_attn = stacked @ self.vh.transpose(-2, -1)           # [nh, T, T]
        grad_vh   = self.attn.transpose(-2, -1) @ stacked         # [nh, T, d_k]

        # 5) Softmax backward
        sum_grad    = (grad_attn * self.attn).sum(dim=-1, keepdim=True)
        grad_scores = self.attn * (grad_attn - sum_grad)          # [nh, T, T]
        if self.mask is not None:
            grad_scores = grad_scores.masked_fill(self.mask == 0, 0)

        # 6) Scaled-dot backward
        factor  = 1.0 / math.sqrt(self.d_k)
        grad_qh = grad_scores @ self.kh * factor                 # [nh, T, d_k]
        grad_kh = grad_scores.transpose(-2, -1) @ self.qh * factor  # [nh, T, d_k]

        # 7) Combine heads
        grad_q = self.combine_heads(grad_qh)                      # [T, D]
        grad_k = self.combine_heads(grad_kh)
        grad_v = self.combine_heads(grad_vh)

        # 8) Assign to linear projections
        self.linear_q.grad = grad_q
        self.linear_k.grad = grad_k
        self.linear_v.grad = grad_v

        # 9) Backprop through Q/K/V projections
        self.linear_v.backward()
        self.linear_k.backward()
        self.linear_q.backward()




    def step(self, learning_rate):
        self.linear_q.step(learning_rate= learning_rate)
        self.linear_k.step(learning_rate= learning_rate)
        self.linear_v.step(learning_rate= learning_rate)
        self.linear_out.step(learning_rate= learning_rate)
        self.W_k.step(learning_rate= learning_rate)
        self.W_q.step(learning_rate= learning_rate)
        self.W_v.step(learning_rate= learning_rate)
        self.W_o.step(learning_rate= learning_rate)
        self.b_k.step(learning_rate= learning_rate)
        self.b_q.step(learning_rate= learning_rate)
        self.b_v.step(learning_rate= learning_rate)
        self.b_o.step(learning_rate= learning_rate)


    def clear_grad(self):
        self.linear_q.clear_grad()
        self.linear_k.clear_grad()
        self.linear_v.clear_grad()
        self.linear_out.clear_grad()
        self.concat.clear_grad()
        self.W_k.clear_grad()
        self.W_q.clear_grad()
        self.W_v.clear_grad()
        self.W_o.clear_grad()
        self.b_k.clear_grad()
        self.b_q.clear_grad()
        self.b_v.clear_grad()
        self.b_o.clear_grad()
        

