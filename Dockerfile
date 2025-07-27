FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so the container can work fully offline
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from transformers import pipeline
pipeline('summarization', model='t5-small')
PY

COPY . .



CMD ["python", "-m", "src.main", "--input", "/input", "--output", "/output/challenge1b_output.json"]
