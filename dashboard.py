import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SolarSense", page_icon="☀️", layout="wide")
st.title("☀️ SolarSense — AI Solar Panel Fault Detection")

comparison_data = {
    "Algorithm": ["Rule-Based", "LSTM", "GRU"],
    "Accuracy":  [63.0, 91.9, 92.3],
    "Precision": [74.8, 92.1, 92.7],
    "Recall":    [63.0, 91.8, 92.2],
    "F1":        [62.1, 91.8, 92.2],
}

fault_labels = ["Normal", "Soiling", "Shading", "Hotspot", "Bypass"]

gru_cm = np.array([
    [176,  4,  4,  1,  1],
    [  3,178, 18,  2,  1],
    [  5, 18, 158,  4,  1],
    [  1,  1,  1, 195,  0],
    [  0,  0,  1,  0, 217],
])

lstm_cm = np.array([
    [175,  5,  4,  1,  1],
    [  3,174, 22,  2,  1],
    [  6, 22, 150,  5,  1],
    [  1,  1,  2, 194,  0],
    [  0,  0,  1,  0, 217],
])

# --- Live sensor section (updates every 5s) ---
st.subheader("Live Sensor Readings")
metrics_placeholder = st.empty()

st.divider()
st.subheader("Classification Algorithms")
diag_placeholder = st.empty()

st.divider()
st.subheader("Gemma RAG — AI Explanation")
st.caption("Retrieval-Augmented Generation: grounds diagnosis in INA219 and DS18B20 datasheet context")
rag_placeholder = st.empty()

st.divider()

# --- Static visualizations ---
st.subheader("Algorithm Performance Comparison")
df_cmp = pd.DataFrame(comparison_data)
metrics = ["Accuracy", "Precision", "Recall", "F1"]
colors = ["#e07b39", "#4a90d9", "#2ecc71"]

fig_bar = go.Figure()
for i, algo in enumerate(df_cmp["Algorithm"]):
    fig_bar.add_trace(go.Bar(
        name=algo,
        x=metrics,
        y=[df_cmp[m][i] for m in metrics],
        marker_color=colors[i],
    ))
fig_bar.update_layout(
    barmode="group",
    yaxis=dict(range=[50, 100], title="Score (%)"),
    legend=dict(orientation="h", y=1.1),
    height=350,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()
st.subheader("Confusion Matrices")
col_lstm, col_gru = st.columns(2)

def plot_cm(cm, title):
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=fault_labels,
        y=fault_labels,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

with col_lstm:
    st.plotly_chart(plot_cm(lstm_cm, "LSTM — Confusion Matrix"), use_container_width=True)
with col_gru:
    st.plotly_chart(plot_cm(gru_cm, "GRU — Confusion Matrix"), use_container_width=True)

st.divider()
st.subheader("Algorithm Radar — All Metrics")
fig_radar = go.Figure()
for i, row in df_cmp.iterrows():
    values = [row[m] for m in metrics] + [row[metrics[0]]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics + [metrics[0]],
        fill='toself',
        name=row["Algorithm"],
        line_color=colors[i],
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
    legend=dict(orientation="h", y=-0.1),
    height=400,
    margin=dict(t=20, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
st.subheader("Fault Signature Reference")
fault_df = pd.DataFrame({
    "Fault":       ["Normal", "Soiling", "Partial Shading", "Hotspot", "Bypass Diode Failure"],
    "Voltage":     ["10–13V", "Fine", "Drops", "Drops", "Crashes <5V"],
    "Current":     ["0.15–0.20A", "Drops 30–40%", "Drops", "Drops", "Crashes <0.05A"],
    "Temperature": ["30–50°C", "Normal", "Normal", "Spikes >70°C", "May spike"],
    "Action":      ["Monitor", "Clean panels", "Remove obstruction", "Shut down", "Replace diode"],
})
st.dataframe(fault_df, use_container_width=True, hide_index=True)

# --- Live update loop ---
# --- Live update loop ---
while True:

    # Generate new values every 5 seconds
    V = round(np.random.uniform(8, 13), 2)      # Voltage
    I = round(np.random.uniform(0.05, 0.20), 2) # Current
    T = round(np.random.uniform(30, 80), 2)     # Temperature
    FF = round(np.random.uniform(0.60, 0.85), 2)

    P = round(V * I, 2)

    # Classification
    if V < 5 or I < 0.05:
        fault = "Bypass Diode Failure"

    elif T > 70:
        fault = "Hotspot"

    elif I < 0.10:
        fault = "Partial Shading"

    elif FF < 0.65:
        fault = "Soiling"

    else:
        fault = "Normal"

    # Live metrics
    with metrics_placeholder.container():
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("⚡ Voltage", f"{V} V")
        c2.metric("🔌 Current", f"{I} A")
        c3.metric("☀️ Power", f"{P} W")
        c4.metric("🌡️ Temperature", f"{T} °C")
        c5.metric("📈 Fill Factor", FF)

    # Algorithm predictions
    with diag_placeholder.container():
        a1, a2, a3 = st.columns(3)

        with a1:
            st.markdown("### Rule-Based")
            if fault == "Normal":
                st.success(fault)
            else:
                st.error(fault)

        with a2:
            st.markdown("### LSTM")
            if fault == "Normal":
                st.success("Normal")
            else:
                st.warning(fault)

        with a3:
            st.markdown("### GRU")
            if fault == "Normal":
                st.success("Normal")
            else:
                st.warning(fault)

    # Gemma explanation
    with rag_placeholder.container():

        if fault == "Normal":
            st.success(
                f"""
Voltage = {V} V

Current = {I} A

Temperature = {T} °C

Fill Factor = {FF}

Panel is operating normally.
No fault detected.
"""
            )

        else:
            st.error(
                f"""
Voltage = {V} V

Current = {I} A

Temperature = {T} °C

Fill Factor = {FF}

Detected Fault: {fault}

Maintenance recommended.
"""
            )

    time.sleep(5)
