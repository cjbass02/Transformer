# transformer/transformer.py
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock

class Transformer(nn.Module):
    """
    decoder only style transformer
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding.weight.requires_grad = True
        # positional encoding layer
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # found this cool layer thing that lets me create a list of layers 
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        # output layer, right before softmax
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.output_layer.weight.requires_grad = True
        self.output_layer.bias.requires_grad = True


    def generate_causal_mask(self, seq_length):
        # Creates a lower triangular matrix (causal mask)
        mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)
        return mask  

    def forward(self, x, mask=None):
        """
        Args:
            x: token inputs
            mask: mask tensor. generated in not provided
        Returns:
            logits: Output logit of the decoder
            attn_weights_all: List of attention weights from each block
        """

        batch_size, seq_length = x.size()
        #mask the atterntion pattern
        if mask is None:
            mask = self.generate_causal_mask(seq_length).to(x.device)
        #embed the tokens and scale the embeddings
        # the scaling is to prevent the gradients from exploding
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        # add positional encodings
        x = self.pos_encoding(x)

        # pass through the decoder blocks
        # this is a list of attention weights from each block
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all.append(attn_weights)

        # pass through the output layer
        logits = self.output_layer(x)
        return logits, attn_weights_all
