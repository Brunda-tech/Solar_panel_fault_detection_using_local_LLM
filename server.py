from flask import Flask, request, jsonify
import requests
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__)
latest_data = {}
latest_diagnosis = {}

# --- Rule-Based ---
def rule_based_diagnosis(V, I, P, T):
    if V < 5 and I < 0.05 and P < 0.3:
        return "Bypass Diode Failure"
    if V < 10 and I < 0.12 and T >= 70:
        return "Hotspot"
    if V < 10 and I < 0.12 and T < 70:
        return "Partial Shading"
    if I < 0.09 and V >= 10 and T < 70:
        return "Soiling"
    return "Normal"

# --- LSTM Model --- updated architecture
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=24, batch_first=True)
        self.dropout = nn.Dropout(0.40)
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 5)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out)))

# --- GRU Model ---
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

LABELS = ["Normal", "Soiling", "Partial Shading", "Hotspot", "Bypass Diode Failure"]

# --- Train on startup ---
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print("Training LSTM and GRU on startup...")
df = pd.read_csv("solar_data.csv")
X = df[["V", "I", "P", "T", "FF"]].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_t = torch.FloatTensor(X_scaled).unsqueeze(1)
y_t = torch.LongTensor(y)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

def train_model(model, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

lstm_model = train_model(LSTMModel(), epochs=12)
gru_model  = train_model(GRUModel(),  epochs=20)
print("Models ready.")

def predict(model, V, I, P, T, FF):
    row = scaler.transform([[V, I, P, T, FF]])
    tensor = torch.FloatTensor(row).unsqueeze(1)
    with torch.no_grad():
        pred = torch.argmax(model(tensor), dim=1).item()
    return LABELS[pred]

# --- Gemma RAG ---
def rag_diagnosis(data):
    prompt = f"""
You are a solar panel fault detection AI for the SolarSense project.
Sensor readings:
- Voltage: {data['V']}V (normal: 10-13V)
- Current: {data['I']}A (normal: 0.15-0.20A)
- Power: {data['P']}W (normal: 1.5-2.4W)
- Temperature: {data['T']}°C (normal: 30-50°C)
- Fill Factor: {data['FF']} (normal: 0.70-0.78)

Fault signatures:
- Soiling: current drops 30-40%, voltage fine, temp normal
- Partial shading: both V and I drop, temp normal
- Hotspot: V and I drop AND temp spikes above 70°C
- Bypass diode failure: sudden simultaneous crash in V and I
- Low light: everything proportionally low, not a fault

Respond with: 1) Normal or fault 2) Fault type 3) Why in one sentence 4) Recommended action
"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "gemma4", "prompt": prompt, "stream": False}
    )
    return response.json()['response']

# --- Flask Routes ---
@app.route('/data', methods=['POST'])
def receive_data():
    global latest_data, latest_diagnosis
    d = request.json
    latest_data = d
    V, I, P, T, FF = d['V'], d['I'], d['P'], d['T'], d['FF']
    latest_diagnosis = {
        "rule_based": rule_based_diagnosis(V, I, P, T),
        "lstm":       predict(lstm_model, V, I, P, T, FF),
        "gru":        predict(gru_model,  V, I, P, T, FF),
        "gemma_rag":  rag_diagnosis(d)
    }
    return jsonify({"status": "ok", "diagnosis": latest_diagnosis})

@app.route('/latest', methods=['GET'])
def get_latest():
    return jsonify({"data": latest_data, "diagnosis": latest_diagnosis})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
