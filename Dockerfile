FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    LANCEDB_PATH=/app/data/lancedb

WORKDIR /app

# CPU-only torch first; otherwise sentence-transformers pulls the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container never downloads at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# From here on (seed step + runtime) never call the Hugging Face Hub; the model is already cached.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY seed ./seed

# The agronomy notes are static, so the container gets its own LanceDB copy at build time.
RUN python seed/seed_lancedb.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
