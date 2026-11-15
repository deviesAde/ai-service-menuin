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

# 1. Install PyTorch versi CPU khusus (hemat ~1.5 GB dibanding versi default CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Copy & install dependencies lainnya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-download model MobileNetV3 ke image agar saat container jalan langsung siap 100% (offline-ready)
RUN python -c "import torchvision.models as models; models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)"

# 4. Copy seluruh source code aplikasi
COPY . .

# Buat folder uploads jika belum ada
RUN mkdir -p uploads

# Expose port Streamlit (8501) dan FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Jalankan Streamlit
ENTRYPOINT ["streamlit", "run", "app.py"]
