# transformer/positional_encoding.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    implements the trigonometric positional encoding, like in the attention paper.)
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args: max_len: maximum number of sequence length
              d_model: dimension of the model (embedding size)
        """
        # since we are extending the nn.Module class, we need to call the parent class init
        super().__init__()

        # create the positional encoding matrix. It should be be of shape max_len by d_model
        pe = torch.zeros(max_len, d_model)


        # Quick efficiency hack I found: By computing div term first for every scaling factor, you can speed up efficiency by not having to do division in sin and cos
        # This gives you the same formula as in the paper in the end
        # Div term is a tensor of all the scaling factors

        # Create a 2d tensor (unsqueeze) of values 0 - maxlen-1 (arange) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # First make a tensor of all even values up to the model dim, (only need evens because the cos frequency compojent as the sin, so you can resuse it)
        # Create the constant scaling factor (pos/100000^(2i/d_model)). 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        #For the even indices use sin of the postiion 
        pe[:, 0::2] = torch.sin(position * div_term)
        # For the odd indices use cos of the postiion 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        # In the parent class, if we reguster this "buffer", it wont be updated during training, but will be saved state
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model). This is an embedding for a toeken
        Returns:
            Same embedding but with positional encoding added.
        """
        # Get the seqeuence length of the input tensor. This is the number of tokens (or positions) in the input sequence (aka a sentance)
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length]