# ml_pipeline.py
"""
Machine Learning Pipeline for Candlestick Pattern Recognition

Steps:
 1. Fetch and preprocess data
 2. Generate sequences and labels
 3. Split into training/validation sets
 4. Train the PatternNN model
 5. Evaluate on validation set (accuracy, confusion matrix)
 6. Save trained model and metrics
"""
import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import List, Tuple, Dict

from etrade_candlestick_bot import ETradeClient, PatternNN
from performance_utils import get_candles_cached
from model_manager import save_model


def prepare_dataset(
    client: ETradeClient,
    symbols: List[str],
    seq_len: int = 10,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fetch candlestick data and build sequence dataset with labels.
    Uses rule-based detectors for labels:
      0: Hammer, 1: Bullish Engulfing, 2: Bearish Engulfing,
      3: Doji, 4: Morning Star, 5: Evening Star

    Returns X_train, X_val, y_train, y_val as Tensors.
    """
    X, y = [], []
    pattern_funcs = [
        lambda seq: seq,  # placeholder; must implement label logic here
    ]  # TODO: replace with actual detector list

    # Fetch and process each symbol
    for sym in symbols:
        df = client.get_candles(sym, interval="5min", days=5)
        values = df[['open','high','low','close','volume']].values
        for i in range(seq_len, len(values)):
            seq = values[i-seq_len:i]
            # TODO: implement label extraction using CandlestickPatterns
            label = 0
            X.append(seq)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    return X_train, X_val, y_train, y_val


def train_and_evaluate(
    client: ETradeClient,
    symbols: List[str],
    model: PatternNN,
    seq_len: int = 10,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_path: str = 'models/pattern_nn.pth',
    metrics_path: str = 'models/metrics.json'
) -> Dict[str, float]:
    """
    Full pipeline: prepare data, train model, evaluate, and save.
    """
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_dataset(client, symbols, seq_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{epochs} Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        acc = accuracy_score(y_val.numpy(), preds)
        cm = confusion_matrix(y_val.numpy(), preds)

    metrics = {'accuracy': acc, 'confusion_matrix': cm.tolist()}

    # Save model and metrics
    save_model(model, model_path, metadata={'accuracy': acc, 'epochs': epochs, 'seq_len': seq_len})
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    # Environment variables or config
    client = ETradeClient(
        consumer_key=os.getenv('ETRADE_CONSUMER_KEY', ''),
        consumer_secret=os.getenv('ETRADE_CONSUMER_SECRET', ''),
        oauth_token=os.getenv('ETRADE_OAUTH_TOKEN', ''),
        oauth_token_secret=os.getenv('ETRADE_OAUTH_TOKEN_SECRET', ''),
        account_id=os.getenv('ETRADE_ACCOUNT_ID', ''),
        sandbox=True
    )
    symbols = os.getenv('SYMBOLS', 'AAPL,MSFT').split(',')
    model = PatternNN()
    metrics = train_and_evaluate(
        client, symbols, model,
        seq_len=10, epochs=5, batch_size=32, lr=1e-3,
        model_path='models/pattern_nn_v1.pth',
        metrics_path='models/pattern_nn_metrics.json'
    )
    print("Training complete. Metrics:", metrics)
