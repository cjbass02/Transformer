# transformer/multihead_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    My multi head attention, modeled after the attention paper.
    """
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: embedding size
            num_heads: number of attention heads.
        """

        # since we are extending the nn.Module class, we need to call the parent class init
        super().__init__()
        # quick sanity check
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        # per head dimension sizwe
        self.d_k = d_model // num_heads

        # Learnable linear layers for queries, keys, and values
        # These should all be of size d_model by d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_q.weight.requires_grad = True
        self.linear_q.bias.requires_grad = True

        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_k.weight.requires_grad = True
        self.linear_k.bias.requires_grad = True

        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_v.weight.requires_grad = True
        self.linear_v.bias.requires_grad = True
        # create a final linear layer for the post-concat output of the attention heads (last node in the multi head attaention diagram)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_out.weight.requires_grad = True
        self.linear_out.bias.requires_grad = True


    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # mask is a boolean tensor showing which entries to zero out in the attention pattern table

        # matmul query and key. This creates that large atterntion pattern table. Each entry gives wirght to how relevant the word in the query is to the word in the key?
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Mask the vlaues to avoid attending to future tokens
        if mask is not None:
            # masked fill just fills the values of the attention heads with -1e9
            scores = scores.masked_fill(mask == 0, -1e9)
        #Get softmax scores from the attention pattern table. 
        attn = F.softmax(scores, dim=-1)
        output = attn @ value
        return output, attn

    def forward(self, query, key=None, value=None, mask=None):
        """
        Args:
            query: Tensor of shape (batch_size, seq_length, d_model)
            mask: mask tensor boolean
            key: Tensor of shape (batch_size, seq_length, d_model)
            value: Tensor of shape (batch_size, seq_length, d_model)
            key and value are optional, if not provided, they are assumed to be equal to query
        Returns:
            (output, attention weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_length, _ = query.size()

        # get the query, key, and values from the projection of inputs 
        # The view function chnages the shape of thr projected input to (batch_size, seq_length, num_heads, d_k) This bassically split the d_model dimension into two dimensions: num_heads and number of dimensions per head (d_k)
        # We need to reshape these tensors so each head can opperate by itself in its own "subspace"
        # Also, by doing this, when we make one call to the scaled dot attetnion, the function essentially batches the heads
        q = self.linear_q(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # apply the scaled dot product attention
        attn_output, attn = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate heads, we need to get the output back to the origial shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        # last linear projection
        output = self.linear_out(attn_output)
        # return the attn too for interpretability later on
        return output, attn
