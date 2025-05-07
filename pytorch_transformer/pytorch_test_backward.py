# main.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_transformer.transformer import Transformer
from pytorch_transformer.dummy_vocab import dummy_vocab  # our large dummy vocabulary

# Set vocab_size based on our dummy vocabulary.
vocab_size = len(dummy_vocab)

def generate_text(model, prompt_tokens, max_gen_length=10, top_k=1):
    """
    Generate text tokens using the autoregressive decoder-only Transformer.
    Args:
        model: the Transformer model.
        prompt_tokens: list of token indices to start the prompt.
        max_gen_length: number of tokens to generate.
        top_k: number of top tokens to consider (greedy if top_k=1).
    Returns:
        List of token indices (prompt + generated tokens).
    """
    model.eval()
    generated = prompt_tokens.copy()  # start with the prompt
    for _ in range(max_gen_length):
        # Prepare input: shape (1, current_seq_length)
        x = torch.tensor([generated], dtype=torch.long)
        logits, _ = model(x)
        # Get logits for the last token.
        last_logits = logits[:, -1, :]  # shape: (1, vocab_size)
        probs = F.softmax(last_logits, dim=-1)
        # Greedy selection: choose the token with the highest probability.
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        next_token = top_indices[0, 0].item()
        generated.append(next_token)
    return generated

def tokens_to_text(tokens, vocab):
    """
    Convert a list of token indices to a text string using the vocabulary.
    """
    words = [vocab.get(token, "<UNK>") for token in tokens]
    return " ".join(words)

def train_model(model, optimizer, input_data, target_data, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(input_data)
        # For language modeling, targets are the input sequence shifted left by one.
        # Here, since our dummy data is random, we use the entire sequence.
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_data.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

def main():
    # Model hyperparameters (using small values for demonstration)
    d_model = 32       # Embedding dimension.
    num_heads = 4      # Number of attention heads.
    d_ff = 64          # Feed-forward network dimension.
    num_layers = 2     # Number of decoder blocks.
    max_len = 50       # Maximum sequence length.

    # Instantiate the decoder-only Transformer.
    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    
    # Create a dummy training dataset.
    # For this demonstration, we'll create a batch of random token sequences.
    batch_size = 16
    seq_length = 10  # Example sequence length for training
    input_data = torch.randint(1, vocab_size, (batch_size, seq_length), dtype=torch.long)
    # For language modeling, targets are typically the same sequence shifted by one.
    # Here, for simplicity, we use the same sequence as target.
    target_data = input_data.clone()

    # Instantiate the optimizer.
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Generate text from an initial prompt before training.
    prompt_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]  # "once upon time in a land far away there lived a brave knight"
    print("Initial prompt:", tokens_to_text(prompt_tokens, dummy_vocab))
    initial_generated = generate_text(model, prompt_tokens, max_gen_length=10, top_k=1)
    print("Before training, generated text:")
    print(tokens_to_text(initial_generated, dummy_vocab))
    
    print("\nStarting training...\n")
    # Train the model for a few epochs.
    train_model(model, optimizer, input_data, target_data, epochs=200)

    # Generate text from the same prompt after training.
    print("\nAfter training, generated text:")
    trained_generated = generate_text(model, prompt_tokens, max_gen_length=10, top_k=1)
    print(tokens_to_text(trained_generated, dummy_vocab))

if __name__ == '__main__':
    main()
