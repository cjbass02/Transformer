# transformer/feed_forward.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    implements the feed forward netowrks in the paper (section 3.3)
    """
    def __init__(self, d_model = 512, d_ff = 2048):
        """
        Args:
            d_model: Dimensionality of the model
            d_ff: Dimensionality of the feed forward network
            *default values are from the paper* They seem oddly small....
        """
            
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear1.weight.requires_grad = True
        self.linear1.bias.requires_grad = True

        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear2.weight.requires_grad = True
        self.linear2.bias.requires_grad = True

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)
