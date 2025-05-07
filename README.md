# Dual Transformer Implementation
This repo features two GPT-style decoder only transformer APIs. The first, a pytorch supported one which uses autograd to manage backpropagation. The Second, a from scratch version that implements its own logic for **every** layer, including backpropagation, decoder blocks, attention, etc.

It also features a FastAPI API for creating, training, and sentance completion.

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

├── from_scratch_transformer: no autograd
    
    ├── layers.py # Core Layer classes: Input, Linear, ReLU, Sum, Concat, Softmax 
    
    ├── positional_encoding.py # Trigonometric positional encodings 
    
    ├── rms_norm.py # Custom RMSNorm layer (γ scale + backprop) 
    
    ├── feed_forward.py # Two-layer feed-forward network (Linear → ReLU → Linear) 
    
    ├── multihead_attention.py # From-scratch multi-head self-attention (2D inputs) 
    
    ├── decoder_block.py # Single Transformer decoder block (MHA → Norm → FF → Norm) 
    
    ├── transformer.py # Stacked decoder blocks + embedding + output projection 
    
├── pytorch_transformer: uses autograd
    
    ├── positional_encoding.py # Trigonometric positional encodings 
    
    ├── feed_forward.py # Two-layer feed-forward network (Linear → ReLU → Linear) 
    
    ├── multihead_attention.py # multi-head self-attention
    
    ├── decoder_block.py # Single Transformer decoder block (MHA → Norm → FF → Norm) 
    
    ├── transformer.py # Stacked decoder blocks + embedding + output projection 

    ├── encoder_block.py # Encoder stub (works, but commented out) juts in case

    ├── decoder.py #legacy, not used

    ├── dummy_vocab.py # A word bank to test training

    ├── pytorch_test_forward # POC forward pass test on the pytorch transformer

    ├── pytorch_test_backward # POC backwards training test on the pytorch transformer

├── from_scratch_training.ipynb # Training loop test for from_scratch_transformer: Used for training on MSOE's supercomputer.

├── input.txt # Example training data (Shakespeare, etc.) 

├── input_short.txt # Shortened training data, better for testing 

├── transformer_api.py # FastAPI API for training and inference

├── fast_api_test.ipynb # FastAPI test script

---

### Requirements

- **Python 3.8+**  
- **PyTorch** (only for tensor operations, no `torch.nn` or autograd; tested on `torch>=1.9`)

Install via:

```bash
pip install torch
pip install fastapi
pip install uvicorn
```

### FastAPI Setup
```bash
uvicorn transformer_api:app --host 0.0.0.0 --port 8000 --reload
