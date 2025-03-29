# transformer/decoder_block.py
import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward

class DecoderBlock(nn.Module):
    """
    A transformer decoder block. This makes it easy to build fun transformers of different configurations
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # "MAsked" self attention block (first in the diagram)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # # attention block that takes in encoding output (second in the diagram)
        # self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        # feed forward network (third in the diagram)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # norm layers
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm1.weight.requires_grad = True
        self.layernorm1.bias.requires_grad = True

        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm2.weight.requires_grad = True
        self.layernorm2.bias.requires_grad = True
        # self.layernorm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x:input embeddings
            enc_output: encoder output
            tgt_mask: mask for decoder self attention 
            src_mask: mask for encoder attention.
        Returns:
            x: decoder output
            self_attn_weights: attention weights from the masked self attention
        """
        # masked self att
        self_attn_output, self_attn_weights = self.self_attn(x, mask=tgt_mask)
        # norm 1
        x = self.layernorm1(x + self_attn_output)
        
        # # encoder attention, x is query, but now we need to pass in the output of the encoder as key and value.
        # enc_dec_attn_output, enc_dec_attn_weights = self.enc_dec_attn(x, key=enc_output, value=enc_output, mask=src_mask)
        # x = self.layernorm2(x + enc_dec_attn_output)
        
        # Feed-forward network.
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + ff_output)
        
        return x, self_attn_weights
