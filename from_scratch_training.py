import torch
from from_scratch_transformer.transformer import Transformer
from from_scratch_transformer.layers import Input, Softmax
# from_scratch_training.py
# Hyperparameters
seq_len       = 100    # sequence length
epochs        = 10     # number of epochs
learning_rate = 1e-3   # learning rate

# Model dimensions
d_model    = 512
num_heads  = 8
d_ff       = 2048
num_layers = 6

# 1) Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2) Build vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# 3) Encode full text as token IDs
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

def get_batch(start):
    """Return a single sequence chunk of length seq_len."""
    x = data[start : start + seq_len]            # [T]
    y = data[start + 1 : start + 1 + seq_len]    # [T]
    return x, y

# 4) Initialize model
model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers)

# 5) Training loop
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    steps = 0

    for i in range(0, data.size(0) - seq_len - 1, seq_len):
        x_seq, y_seq = get_batch(i)  # each is [T]

        # Forward pass (no batch dim)
        out_lin = model.forward(x_seq)      # out_lin.output: [T, V]
        logits  = out_lin.output            # [T, vocab_size]
        T, V    = logits.shape

        # Prepare one-hot targets
        logits_flat    = logits           # already [T, V]
        target_onehot  = torch.zeros_like(logits_flat)
        target_onehot[torch.arange(T), y_seq] = 1.0
        target_layer   = Input(T, V)
        target_layer.set(target_onehot)

        # Compute loss
        loss_layer = Softmax(out_lin, target_layer)
        loss_layer.forward()
        loss = loss_layer.output            # scalar tensor

        # Backward pass
        loss_layer.backward()
        model.backward()

        # Update parameters (now accepts learning_rate)
        model.step(learning_rate)

        # Clear gradients
        loss_layer.clear_grad()
        model.clear_grad()

        total_loss += loss.item()
        steps += 1
        if steps % 10 == 0:
            print(f'Epoch {epoch}/{epochs} — Step {steps} — Loss: {loss.item():.4f}')

    avg_loss = total_loss / steps if steps > 0 else float('nan')
    print(f'Epoch {epoch}/{epochs} — avg_loss: {avg_loss:.4f}')
