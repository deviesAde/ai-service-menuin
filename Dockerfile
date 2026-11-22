# Gunakan base image Python 3.11 slim yang ringan
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install dependensi sistem minimal untuk OpenCV & Media
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# 1. Install dependencies (ONNX Runtime ringan tanpa PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Pre-download model MobileCLIP-S0 ONNX ke image agar container langsung siap 100% (offline-ready)
RUN mkdir -p models && \
    curl -fL -o models/mobileclip_s0_vision.onnx \
      https://huggingface.co/Xenova/mobileclip_s0/resolve/main/onnx/vision_model.onnx

# 3. Copy seluruh source code aplikasi
COPY . .

# Buat folder uploads jika belum ada
RUN mkdir -p uploads

# Expose port Streamlit (8501) dan FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Jalankan FastAPI secara default (atau override via docker-compose untuk Streamlit)
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
