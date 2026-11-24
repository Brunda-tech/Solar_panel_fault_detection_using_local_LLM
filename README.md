# ☀️ SolarSense – AI Solar Panel Fault Detection

### Real-Time Solar Panel Fault Detection Using Machine Learning and Local LLM

SolarSense is an AI-powered solar panel monitoring and fault detection system that combines IoT hardware, machine learning algorithms, and a Local Large Language Model (LLM) to identify solar panel faults and provide intelligent diagnostic explanations.

The system continuously monitors solar panel parameters such as voltage, current, power, temperature, and fill factor using ESP32 and sensor modules. It applies Rule-Based Classification, LSTM, and GRU models to detect different fault conditions and uses a Gemma-powered RAG diagnostic agent to explain the detected faults in natural language.

## 🚀 Key Features

### ⚡ Real-Time Solar Panel Monitoring
- Collects real-time solar panel readings using hardware sensors.
- Measures voltage and current using the INA219 sensor.
- Monitors temperature using the DS18B20 temperature sensor.
- Calculates power and fill factor.
- Transmits sensor readings through ESP32 using Wi-Fi communication.

### 🔍 AI-Based Fault Detection

The system identifies different solar panel operating conditions:

| Condition | Description |
|---|---|
| Normal | Solar panel operates under normal conditions. |
| Soiling | Dust and dirt accumulation reduces panel efficiency. |
| Partial Shading | Partial obstruction reduces solar power generation. |
| Hotspot | Abnormal temperature increase in solar panel cells. |
| Bypass Diode Failure | Failure of bypass diode affects panel voltage and current output. |

### 🤖 Machine Learning Algorithms

The project implements three classification approaches:

**1. Rule-Based Classification**
- Uses predefined threshold conditions.
- Identifies faults using voltage, current, and temperature readings.
- Provides fast and explainable predictions.

**2. LSTM (Long Short-Term Memory)**
- Uses recurrent neural network architecture.
- Learns patterns from sequential sensor readings.
- Helps identify abnormal solar panel operating conditions.

**3. GRU (Gated Recurrent Unit)**
- Uses a lightweight recurrent neural network architecture.
- Learns temporal relationships in sensor data.
- Provides efficient fault classification.

### 🧠 Local LLM – Gemma RAG Diagnostic Agent

- Integrates a locally running Gemma model through Ollama.
- Uses Retrieval-Augmented Generation (RAG).
- Retrieves relevant technical information from sensor datasheets.
- Generates natural-language explanations for detected faults.
- Provides possible causes and recommended corrective actions.

### 📊 Interactive Dashboard

The Streamlit dashboard provides:

- Live voltage, current, power, temperature, and fill factor readings.
- Real-time fault classification results.
- Comparison of Rule-Based, LSTM, and GRU algorithms.
- Model performance metrics.
- Confusion matrices.
- Performance comparison charts.
- AI-generated fault explanations.

## 🏗️ System Architecture

The project follows this processing pipeline:

```text
       SOLAR PANEL ARRAY (2S2P)
                  |
                  v
       INA219 + DS18B20 Sensors
                  |
                  v
           ESP32 DevKit V1
                  |
                  v
         Wi-Fi Communication
                  |
                  v
            Flask Server
                  |
         +--------+--------+
         |        |        |
         v        v        v
      Rule-Based  LSTM     GRU
      Classifier  Model   Model
         |        |        |
         +--------+--------+
                  |
                  v
          Gemma RAG Agent
           (Ollama + FAISS)
                  |
                  v
         Streamlit Dashboard
                  |
                  v
       Fault Detection Results
       & AI Diagnostic Reports
```

## 🛠️ Hardware Components

| Component | Purpose |
|---|---|
| ESP32 DevKit V1 | Microcontroller and Wi-Fi communication |
| INA219 Sensor | Voltage and current measurement |
| DS18B20 Sensor | Temperature measurement |
| Mini Solar Panel Array (2S2P) | Solar power generation |
| 4.7 kΩ Resistor | DS18B20 pull-up resistor |
| 111 Ω Load Resistor | Electrical load |
| Breadboard and Jumper Wires | Hardware connections |

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Flask | Backend server and sensor data handling |
| Streamlit | Interactive monitoring dashboard |
| PyTorch | Deep learning model development |
| LSTM | Sequential data classification |
| GRU | Recurrent neural network classification |
| Scikit-learn | Data preprocessing and model evaluation |
| Pandas | Dataset handling and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| Ollama | Local LLM execution |
| Gemma | AI-powered fault explanation |
| FAISS | Vector similarity search |
| ESP32 | IoT hardware integration |
| INA219 & DS18B20 | Sensor data acquisition |

## 📈 Model Performance Comparison

The project evaluates three classification approaches using multiple performance metrics.

| Evaluation Metric | Rule-Based | LSTM | GRU |
|---|---:|---:|---:|
| Accuracy | 63.0% | 91.9% | 92.3% |
| Precision | 74.8% | 92.1% | 92.7% |
| Recall | 63.0% | 91.8% | 92.2% |
| F1-Score | 62.1% | 91.8% | 92.2% |

**Key Observation:** The GRU model achieved the highest accuracy of 92.3%, demonstrating better classification performance than the Rule-Based and LSTM approaches on the evaluated dataset.

## 📂 Project Structure

```text
SolarSense/
│
├── server.py
│   └── Flask backend and fault detection integration
│
├── dashboard.py
│   └── Streamlit monitoring dashboard
│
├── rule_based.py
│   └── Rule-based fault classification
│
├── lstm_model.py
│   └── LSTM deep learning model
│
├── gru_model.py
│   └── GRU deep learning model
│
├── compare.py
│   └── Algorithm performance comparison
│
├── rag.py
│   └── Gemma RAG diagnostic agent
│
├── solar_data.csv
│   └── Solar panel sensor dataset
│
├── ina219.pdf
│   └── INA219 sensor datasheet
│
├── DS18B20.pdf
│   └── DS18B20 sensor datasheet
│
└── README.md
    └── Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Brunda-tech/Solar_panel_fault_detection_using_local_LLM.git
```

### 2. Navigate to the Project Directory

```bash
cd Solar_panel_fault_detection_using_local_LLM
```

### 3. Install Required Dependencies

```bash
pip install flask requests streamlit plotly pandas numpy
```

```bash
pip install torch scikit-learn
```

```bash
pip install pypdf sentence-transformers faiss-cpu
```

### 4. Install Ollama

Install Ollama and configure the Gemma model for local execution.

Download: https://ollama.com/

Pull the required model:

```bash
ollama pull gemma4
```

### 5. Start the Ollama Server

```bash
ollama serve
```

### 6. Run the Flask Backend

```bash
python server.py
```

### 7. Launch the Streamlit Dashboard

Open another terminal:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

## 📡 Hardware Communication

The ESP32 collects sensor readings and transmits them to the Flask backend through Wi-Fi.

The transmitted parameters include:

- Voltage (V)
- Current (I)
- Power (P)
- Temperature (T)
- Fill Factor (FF)

The system processes incoming sensor readings and generates fault classification results.

## 🔮 Future Enhancements

The system can be further improved by introducing:

- Cloud-based solar panel monitoring.
- Mobile application integration.
- Predictive maintenance using historical sensor data.
- Solar panel efficiency optimization.
- Automated fault notifications and alerts.
- Support for larger solar panel installations.
- Advanced AI-based fault severity analysis.
- Improved local LLM performance and optimization.

## 🎯 Project Outcomes

- Developed an IoT-based solar panel monitoring system.
- Integrated ESP32 with INA219 and DS18B20 sensors.
- Implemented real-time sensor data communication.
- Developed and compared Rule-Based, LSTM, and GRU classification approaches.
- Achieved 92.3% classification accuracy using the GRU model on the evaluated dataset.
- Integrated a Local LLM-based RAG diagnostic agent.
- Developed an interactive dashboard for monitoring and fault analysis.

## 👥 Team Members

This project was collaboratively developed by:

| Team Member | GitHub Profile |
|---|---|
| **Brunda S** | [@Brunda-tech](https://github.com/Brunda-tech) |
| **Chandhan Kumar** | [@chandhankumargv](https://github.com/chandhankumargv) |

## 🤝 Collaboration

We collaboratively developed this project by combining IoT hardware integration, machine learning, deep learning, and Generative AI to build an intelligent solar panel fault detection and diagnostic system.

## 📜 License

This project is developed for educational and research purposes.
