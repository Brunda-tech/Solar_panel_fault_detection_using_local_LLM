import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import requests

# --- Load and chunk PDFs ---
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

print("Loading datasheets...")
ina219_text  = load_pdf("ina219.pdf")
ds18b20_text = load_pdf("DS18B20.pdf")

all_chunks = chunk_text(ina219_text) + chunk_text(ds18b20_text)
print(f"Total chunks: {len(all_chunks)}")

# --- Embed chunks ---
print("Embedding chunks...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(all_chunks, show_progress_bar=True)

# --- Build FAISS index ---
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("FAISS index built.")

# --- RAG Diagnosis Function ---
def rag_diagnosis(V, I, P, T, FF, top_k=3):
    query = f"Solar panel reading: Voltage={V}V, Current={I}A, Power={P}W, Temperature={T}C, FillFactor={FF}"
    query_vec = embedder.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    context = "\n\n".join([all_chunks[i] for i in indices[0]])

    prompt = f"""You are a solar panel fault detection AI.
Use the following datasheet context to inform your diagnosis:

{context}

Sensor readings:
- Voltage: {V}V (normal: 10-13V)
- Current: {I}A (normal: 0.15-0.20A)
- Power: {P}W (normal: 1.5-2.4W)
- Temperature: {T}°C (normal: 30-50°C)
- Fill Factor: {FF} (normal: 0.70-0.78)

Respond with: 1) Normal or fault 2) Fault type 3) Why in one sentence 4) Recommended action
"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "gemma4", "prompt": prompt, "stream": False}
    )
    return response.json()['response']

# --- Test ---
test_cases = [
    (11.5, 0.18, 2.07, 40,  0.74, "Normal"),
    (11.2, 0.08, 0.90, 42,  0.71, "Soiling"),
    (8.5,  0.10, 0.85, 38,  0.69, "Partial Shading"),
    (8.0,  0.09, 0.72, 85,  0.68, "Hotspot"),
    (3.0,  0.02, 0.06, 45,  0.60, "Bypass Diode Failure"),
]

print("\n--- RAG Diagnosis Results ---")
for V, I, P, T, FF, expected in test_cases:
    print(f"\nExpected: {expected}")
    print("Gemma says:", rag_diagnosis(V, I, P, T, FF))
    print("-" * 40)
