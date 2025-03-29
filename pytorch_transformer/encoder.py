# # transformer/encoder.py
# import torch
# import torch.nn as nn
# from .positional_encoding import PositionalEncoding
# from .encoder_block import TransformerEncoderBlock

# class Encoder(nn.Module):
#     """
#     transformer encoder layer
#     """
#     def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, max_len=5000):
#         super().__init__()
#         # nn embedding layer
#         self.embedding = nn.Embedding(input_dim, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_len)
#         # found this cool layer thing that lets me create a list of layers 
#         self.layers = nn.ModuleList([
#             TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
#         ])
        
#     def forward(self, src, mask=None):
#         """
#         Args:
#             src: Source  text
#             mask: boolean matrix
#         Returns:
#             encoder output to be sent to decoder and list of attention weights from each block.
#         """
#         # embed the sorce text
#         x = self.embedding(src)
#         # add positional encodings
#         x = self.positional_encoding(x)
#         #make a list of attention weights 
#         attn_weights_all = []
#         # loop through all layers in the module list
#         for layer in self.layers:
#             x, attn_weights = layer(x, mask)
#             attn_weights_all.append(attn_weights)
#         return x, attn_weights_all
