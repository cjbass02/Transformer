# Dual Transformer Implementation
This repo features two GPT-style decoder only transformer APIs. The first, a pytorch supported one which uses autograd to manage backpropagation. The Second, a from scratch version that implements its own logic for **every** layer, including backpropagation, decoder blocks, attention, etc.


---
## From Scratch Implementation
### Features

- **2D-only tensors**: All layers operate on `[T, D]` sequences—no batch dimension required.  
- **Custom backprop**: Every layer (`Linear`, `ReLU`, `Sum`, `Concat`, `RMSNorm`, `MultiHeadAttention`, `FeedForward`) implements its own `forward()`, `backward()`, and parameter `step()` logic.  
- **Decoder-only architecture**: Support for causal self-attention with optional masking.  
- **RMSNorm**: A simple RMS normalization layer with learnable scale (γ) and custom backward pass.  
- **Training script**: Character-level language modeling on raw text (`input.txt`) with sliding-window teacher forcing.

---

### Repository Structure

├── layers.py # Core Layer classes: Input, Linear, ReLU, Sum, Concat, Softmax 

├── positional_encoding.py # Trigonometric positional encodings 

├── rms_norm.py # Custom RMSNorm layer (γ scale + backprop) 

├── feed_forward.py # Two-layer feed-forward network (Linear → ReLU → Linear) 

├── multihead_attention.py # From-scratch multi-head self-attention (2D inputs) 

├── decoder_block.py # Single Transformer decoder block (MHA → Norm → FF → Norm) 

├── transformer.py # Stacked decoder blocks + embedding + output projection 

├── from_scratch_training.py# Training loop Test: sequences of length seq_len, Softmax loss, 

├── input.txt # Example training data (Shakespeare, etc.) 

---

### Requirements

- **Python 3.8+**  
- **PyTorch** (only for tensor operations, no `torch.nn` or autograd; tested on `torch>=1.9`)

Install via:

```bash
pip install torch
