import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# Load data
df = pd.read_csv("solar_data.csv")
X = df[["V", "I", "P", "T", "FF"]].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Rule-Based ---
def rule_based_diagnosis(V, I, P, T):
    if V < 5 and I < 0.05 and P < 0.3:
        return 4  # Bypass
    if V < 10 and I < 0.12 and T >= 70:
        return 3  # Hotspot
    if V < 10 and I < 0.12 and T < 70:
        return 2  # Shading
    if I < 0.09 and V >= 10 and T < 70:
        return 1  # Soiling
    return 0  # Normal

test_df = df.iloc[pd.Series(range(len(df))).sample(200, random_state=42).values]
X_test_raw = df[["V", "I", "P", "T", "FF"]].values
y_test_raw = df["label"].values

start = time.time()
rb_preds = [rule_based_diagnosis(row[0], row[1], row[2], row[3]) for row in X_test_raw]
rb_latency = (time.time() - start) / len(X_test_raw) * 1000

# --- PyTorch helper ---
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

def train_and_eval(ModelClass, name):
    model = ModelClass()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(20):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    model.eval()
    with torch.no_grad():
        start = time.time()
        preds = torch.argmax(model(X_test_t), dim=1).numpy()
        latency = (time.time() - start) / len(X_test) * 1000
    return preds, train_time, latency

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out)))

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out)))

lstm_preds, lstm_train, lstm_lat = train_and_eval(LSTMModel, "LSTM")
gru_preds,  gru_train,  gru_lat  = train_and_eval(GRUModel,  "GRU")

# --- Print Comparison Table ---
def metrics(y_true, y_pred):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 1),
        "Precision": round(precision_score(y_true, y_pred, average='macro') * 100, 1),
        "Recall":    round(recall_score(y_true, y_pred, average='macro') * 100, 1),
        "F1":        round(f1_score(y_true, y_pred, average='macro') * 100, 1),
    }

rb   = metrics(y_test_raw, rb_preds)
lstm = metrics(y_test, lstm_preds)
gru  = metrics(y_test, gru_preds)

print("\n========== ALGORITHM COMPARISON ==========")
print(f"{'Metric':<12} {'Rule-Based':>12} {'LSTM':>10} {'GRU':>10}")
print("-" * 46)
for key in ["Accuracy", "Precision", "Recall", "F1"]:
    print(f"{key:<12} {rb[key]:>11}% {lstm[key]:>9}% {gru[key]:>9}%")
print("-" * 46)
print(f"{'Latency':<12} {rb_latency:>10.3f}ms {lstm_lat:>8.3f}ms {gru_lat:>8.3f}ms")
print(f"{'Train Time':<12} {'N/A':>12} {lstm_train:>8.1f}s {gru_train:>8.1f}s")
print("==========================================")
