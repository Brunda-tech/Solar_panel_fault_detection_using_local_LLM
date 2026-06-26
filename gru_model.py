import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# Load data
df = pd.read_csv("solar_data.csv")
X = df[["V", "I", "P", "T", "FF"]].values
y = df["label"].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
y_test_t  = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# GRU Model
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
        out = torch.relu(self.fc1(out))
        return self.fc2(out)

model = GRUModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train
start = time.time()
for epoch in range(20):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/20 done")
train_time = time.time() - start

# Evaluate
model.eval()
with torch.no_grad():
    start = time.time()
    y_pred = torch.argmax(model(X_test_t), dim=1).numpy()
    infer_time = (time.time() - start) / len(X_test) * 1000

print("\n--- GRU Results ---")
print(classification_report(y_test, y_pred, target_names=["Normal","Soiling","Shading","Hotspot","Bypass"]))
print(f"Training time: {train_time:.1f}s")
print(f"Inference latency: {infer_time:.2f}ms per sample")
