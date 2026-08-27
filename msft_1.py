import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


import sys
sys.path.append("..")  # so we can import from src/ when running from notebooks/

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_utils import download_stock_data, prepare_data
from src.models import LSTMModel, GRUModel
from src.train_utils import train_model, evaluate_model, plot_predictions

torch.manual_seed(42)
np.random.seed(42)

#

TICKER = "AMZN"

df = download_stock_data(TICKER, start="2010-01-01", cache_dir="data")
df.tail()

plt.figure(figsize=(11, 4))
plt.plot(df.index, df["Close"])
plt.title(f"{TICKER} — Closing Price History")
plt.xlabel("Date")
plt.ylabel("Close price")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/closing_price_history.png", dpi=150)

plt.show()

###
LOOKBACK = 20
TRAIN_SPLIT = 0.8

X_train, y_train, X_test, y_test, scaler = prepare_data(
    df, lookback=LOOKBACK, train_split=TRAIN_SPLIT
)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

print("X_train:", X_train_t.shape, " y_train:", y_train_t.shape)
print("X_test :", X_test_t.shape, " y_test :", y_test_t.shape)


#######
HIDDEN_DIM = 32
NUM_LAYERS = 2
EPOCHS = 100
LR = 0.01

lstm_model = LSTMModel(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1)
gru_model = GRUModel(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1)

lstm_model, gru_model


start = time.time()
lstm_loss_history = train_model(lstm_model, X_train_t, y_train_t, num_epochs=EPOCHS, lr=LR)
lstm_train_time = time.time() - start

print(f"LSTM training time: {lstm_train_time:.2f}s")

start = time.time()
gru_loss_history = train_model(gru_model, X_train_t, y_train_t, num_epochs=EPOCHS, lr=LR)
gru_train_time = time.time() - start

print(f"GRU training time: {gru_train_time:.2f}s")

start = time.time()
gru_loss_history = train_model(gru_model, X_train_t, y_train_t, num_epochs=EPOCHS, lr=LR)
gru_train_time = time.time() - start

print(f"GRU training time: {gru_train_time:.2f}s")


plt.figure(figsize=(9, 4))
plt.plot(lstm_loss_history, label="LSTM train loss")
plt.plot(gru_loss_history, label="GRU train loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss (scaled space)")
plt.title("Training loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/training_loss.png", dpi=150)

plt.show()



lstm_mse, lstm_rmse, lstm_preds, y_test_actual = evaluate_model(lstm_model, X_test_t, y_test_t, scaler)
gru_mse, gru_rmse, gru_preds, _ = evaluate_model(gru_model, X_test_t, y_test_t, scaler)

print(f"{'Model':<6} {'Test MSE':>12} {'Test RMSE':>12} {'Train time (s)':>16}")
print(f"{'LSTM':<6} {lstm_mse:>12.4f} {lstm_rmse:>12.4f} {lstm_train_time:>16.2f}")
print(f"{'GRU':<6} {gru_mse:>12.4f} {gru_rmse:>12.4f} {gru_train_time:>16.2f}")

plot_predictions(
    y_test_actual.flatten(),
    {"LSTM": lstm_preds.flatten(), "GRU": gru_preds.flatten()},
    save_path="results/predictions.png",
)



import json, os

results = {
    "ticker": TICKER,
    "lookback": LOOKBACK,
    "LSTM": {"test_mse": lstm_mse, "test_rmse": lstm_rmse, "train_time_sec": lstm_train_time},
    "GRU": {"test_mse": gru_mse, "test_rmse": gru_rmse, "train_time_sec": gru_train_time},
}

os.makedirs("../results", exist_ok=True)
with open("../results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

results