"""
Training loop, evaluation, and plotting helpers, plus a CLI entry point that
runs the full pipeline: download data -> preprocess -> train LSTM & GRU ->
evaluate -> compare -> save plots/metrics.

Usage:
    python src/train_utils.py --ticker AMZN --lookback 20 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

try:
    # allow running as `python src/train_utils.py`
    from data_utils import prepare_data, download_stock_data
    from models import LSTMModel, GRUModel
except ImportError:  # allow running as `python -m src.train_utils`
    from src.data_utils import prepare_data, download_stock_data
    from src.models import LSTMModel, GRUModel


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose_every: int = 10,
) -> list[float]:
    """Train `model` with MSE loss + Adam. Returns the per-epoch loss history."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if verbose_every and epoch % verbose_every == 0:
            print(f"  epoch {epoch:4d} | train MSE loss: {loss.item():.6f}")

    return loss_history


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, scaler):
    """
    Run inference on the test set and compute MSE/RMSE in original price units
    (i.e. after inverse-transforming the MinMax scaling).
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    y_pred_np = scaler.inverse_transform(y_pred.numpy())
    y_test_np = scaler.inverse_transform(y_test.numpy())

    mse = float(np.mean((y_pred_np - y_test_np) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse, y_pred_np, y_test_np


def plot_predictions(y_test_np, preds_by_model: dict, save_path: str | None = None):
    """Plot actual vs. predicted prices for each model in `preds_by_model`."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    plt.plot(y_test_np, label="Actual", linewidth=2)
    for name, preds in preds_by_model.items():
        plt.plot(preds, label=f"{name} predicted", linestyle="--")
    plt.title("Actual vs. Predicted Closing Price (Test Set)")
    plt.xlabel("Time step (test set)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


def run_pipeline(
    ticker: str = "AMZN",
    lookback: int = 20,
    epochs: int = 100,
    hidden_dim: int = 32,
    num_layers: int = 2,
    lr: float = 0.01,
    train_split: float = 0.8,
    results_dir: str = "results",
):
    print(f"Downloading/loading data for {ticker}...")
    df = download_stock_data(ticker)

    print(f"Preprocessing (lookback={lookback}, train_split={train_split})...")
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, lookback=lookback, train_split=train_split
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    results = {}
    preds_by_model = {}

    for name, ModelClass in [("LSTM", LSTMModel), ("GRU", GRUModel)]:
        print(f"\nTraining {name} model...")
        model = ModelClass(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)

        start = time.time()
        train_model(model, X_train_t, y_train_t, num_epochs=epochs, lr=lr)
        elapsed = time.time() - start

        mse, rmse, y_pred_np, y_test_np = evaluate_model(model, X_test_t, y_test_t, scaler)
        results[name] = {"test_mse": mse, "test_rmse": rmse, "train_time_sec": elapsed}
        preds_by_model[name] = y_pred_np.flatten()

        print(f"  {name} -> test MSE: {mse:.4f} | test RMSE: {rmse:.4f} | train time: {elapsed:.2f}s")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_predictions(
        y_test_np.flatten(),
        preds_by_model,
        save_path=os.path.join(results_dir, "predictions.png"),
    )

    print("\n=== Comparison ===")
    for name, r in results.items():
        print(f"{name}: RMSE={r['test_rmse']:.4f}  train_time={r['train_time_sec']:.2f}s")

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Train & compare LSTM vs GRU for stock price prediction.")
    parser.add_argument("--ticker", type=str, default="AMZN")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        ticker=args.ticker,
        lookback=args.lookback,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        train_split=args.train_split,
        results_dir=args.results_dir,
    )
