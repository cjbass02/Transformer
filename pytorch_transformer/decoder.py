# # transformer/decoder.py
# import torch
# import torch.nn as nn
# from .positional_encoding import PositionalEncoding
# from .decoder_block import DecoderBlock

# class Decoder(nn.Module):
#     """
#     the decoder layer this made up of n_layers decodr blocks
#     """
#     def __init__(self, output_dim, d_model, num_heads, d_ff, num_layers, max_len=5000):
#         super().__init__()
#         #output embeddings
#         self.embedding = nn.Embedding(output_dim, d_model)
#         #postional encoding
#         self.positional_encoding = PositionalEncoding(d_model, max_len)
#         # list of decoder blocks
#         self.layers = nn.ModuleList([
#             DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
#         ])
#         # final linear layer right before softmax in diagram
#         self.output_layer = nn.Linear(d_model, output_dim)
        
#     def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
#         """
#         Args:
#             tgt: target output
#             enc_output: encoder output (kind of like the input here)
#             tgt_mask: Mask for self attention
#             src_mask: Mask for encoder attention
#         Returns:
#             logits: output logits 
#             all_self_attn_weights: self-attention weights from each decoder block
#             all_enc_dec_attn_weights: encoder attention weights from each block
#         """
#         x = self.embedding(tgt)
#         x = self.positional_encoding(x)
        
#         all_self_attn_weights = []
#         all_enc_dec_attn_weights = []
#         for layer in self.layers:
#             x, self_attn_weights, enc_dec_attn_weights = layer(x, enc_output, tgt_mask, src_mask)
#             all_self_attn_weights.append(self_attn_weights)
#             all_enc_dec_attn_weights.append(enc_dec_attn_weights)
            
#         logits = self.output_layer(x)
#         return logits, all_self_attn_weights, all_enc_dec_attn_weights
