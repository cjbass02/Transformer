from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

from from_scratch_transformer.transformer import Transformer
from from_scratch_transformer.layers import Input, Softmax

"""
FastAPI application for a Transformer model.
This API allows users to create a Transformer model, train it on token sequences, and make predictions.
"""



app = FastAPI()
# Global transformer instance
model: Optional[Transformer] = None

# models for request bodies
class CreateModelRequest(BaseModel):
    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    max_len: Optional[int] = 5000

class TrainRequest(BaseModel):
    token_sequences: List[List[int]]
    target_sequences: List[List[int]]
    epochs: int = 1
    learning_rate: float = 1e-3

class PredictRequest(BaseModel):
    input_tokens: List[int]

@app.post("/create")
def create_model(req: CreateModelRequest):
    """
    Create a new Transformer model with the passed in hyperparameters.
    """
    global model
    model = Transformer(
        vocab_size=req.vocab_size,
        d_model=req.d_model,
        num_heads=req.num_heads,
        d_ff=req.d_ff,
        num_layers=req.num_layers,
        max_len=req.max_len or 5000
    )
    return {"message": "Transformer model created"}

@app.post("/train")
def train(req: TrainRequest):
    """
    Train the existing transformer model on provided token sequences
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not created")
    vocab_size = model.output_b.cols
    for epoch in range(req.epochs):
        for tokens, targets in zip(req.token_sequences, req.target_sequences):
            x = torch.tensor(tokens, dtype=torch.long)
            out_lin = model.forward(x)
            # one hot targets
            T = len(targets)
            y = torch.zeros((T, vocab_size))
            for i, t in enumerate(targets):
                y[i, t] = 1
            y_layer = Input(y.shape[0], y.shape[1])
            y_layer.set(y)
            sm = Softmax(out_lin, y_layer)
            sm.forward()
            sm.backward()
            model.step(req.learning_rate)
            model.clear_grad()
    return {"message": f"Training completed for {req.epochs} epochs"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run inference on input token sequence and return logits and predicted next token.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not created")
    x = torch.tensor(req.input_tokens, dtype=torch.long)
    out_lin = model.forward(x)
    logits = out_lin.output.detach().tolist()
    last_logits = torch.tensor(logits[-1])
    predicted = int(torch.argmax(last_logits).item())
    return {"logits": logits, "predicted_token": predicted}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("transformer_api:app", host="0.0.0.0", port=8080, reload=True)




# uvicorn transformer_api:app --host 0.0.0.0 --port 8000 --reload
# http://localhost:8000/docs