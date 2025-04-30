# from_scratch_transformer/transformer.py
from .layers import Input, Linear
from .positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000):
        # Token embedding table (learnable)
        self.embed_weights = Input(vocab_size, d_model)
        self.embed_weights.randomize()

        # Positional encoding layer (no learnable params)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # Stack of decoder blocks
        self.blocks = [DecoderBlock(d_model, num_heads, d_ff)
                       for _ in range(num_layers)]

        #Final linear projection weights & bias
        self.output_w = Input(d_model, vocab_size)
        self.output_w.randomize()
        self.output_b = Input(1, vocab_size)
        self.output_b.randomize()
        self.out_lin = Linear(X=None, W=self.output_w, b=self.output_b)

    def forward(self, tokens):
        """
        tokens: LongTensor of shape [T]
        """
        # Embed and scale
        emb = self.embed_weights.output[tokens] * (self.embed_weights.cols ** 0.5)

        # embeddings into an Input layer
        emb_layer = Input(emb.shape[0], emb.shape[1])
        emb_layer.set(emb)

        # Positional encoding as a layer
        self.pos_enc.x = emb_layer
        self.pos_enc.forward(emb_layer)
        x = self.pos_enc  # x.output is [T, D]

        # Pass through decoder blocks
        for blk in self.blocks:
            x = blk.forward(x)

        # Final projection to vocab logits
        self.out_lin = Linear(X=x, W=self.output_w, b=self.output_b)
        self.out_lin.forward()
        return self.out_lin

    def backward(self):
        # Backprop through final projection
        self.out_lin.backward()
        # Backprop through decoder blocks
        for blk in reversed(self.blocks):
            blk.backward()
        # Backprop through positional encoding into embeddings
        self.pos_enc.backward()


    def step(self, learning_rate):
        # update weights
        self.embed_weights.step(learning_rate= learning_rate)
        self.output_w.step(learning_rate= learning_rate)
        self.output_b.step(learning_rate= learning_rate)
        self.out_lin.step(learning_rate= learning_rate)
        for blk in self.blocks:
            blk.step(learning_rate= learning_rate)

    def clear_grad(self):
        self.embed_weights.clear_grad()
        self.output_w.clear_grad()
        self.output_b.clear_grad()
        self.out_lin.clear_grad()
        for blk in self.blocks:
            blk.clear_grad()

