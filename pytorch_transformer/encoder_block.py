# # transformer/encoder_block.py
# import torch
# import torch.nn as nn
# from .multihead_attention import MultiHeadAttention
# from .feed_forward import FeedForward

# class TransformerEnciderBlock(nn.Module):
#     """
#     A single transformer block in the paper. USed for the encoder. I included this because it is used in the encoder layer. You said "he code should include all of the key components discussed in the readings"
#     so this felt appropriate to include
#     """
#     def __init__(self, d_model, num_heads, d_ff):
#         super().__init__()
#         self.attention = MultiHeadAttention(d_model, num_heads)
#         # the layer we dont know
#         self.layernorm1 = nn.LayerNorm(d_model)
#         self.feed_forward = FeedForward(d_model, d_ff)
#         self.layernorm2 = nn.LayerNorm(d_model)

#     def forward(self, x, mask=None):
#         # make the mutli head attn layer outputs of the sequence x
#         attn_output, attn_weights = self.attention(x, mask)
#         # norm layer
#         x = self.layernorm1(x + attn_output)
#         # feedforward layer
#         ff_output = self.feed_forward(x)
#         #nor mlayer 2
#         x = self.layernorm2(x + ff_output)
#         return x, attn_weights
