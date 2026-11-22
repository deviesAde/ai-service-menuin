import os
import sys
import time
import base64
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Ensure backend directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.feature_extractor import FeatureExtractor

app = FastAPI(
    title="Menuin AI Visual Scanner Service",
    description="Microservice untuk ekstraksi fitur visual kemasan produk (MobileCLIP-S0 ONNX 512-dim)",
    version="2.0.0"
)

# Enable CORS for Next.js web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Feature Extractor
print("[FastAPI] Inisialisasi FeatureExtractor MobileCLIP-S0 (ONNX)...")
extractor = FeatureExtractor()
print("[FastAPI] AI Engine siap menerima permintaan!")

class Base64ImageRequest(BaseModel):
    image_base64: str

class ExtractResponse(BaseModel):
    success: bool
    dimension: int
    inference_ms: float
    embedding: List[float]

@app.get("/")
def root():
    return {
        "service": "Menuin AI Visual Scanner",
        "status": "online",
        "version": "2.0.0",
        "model": extractor.MODEL_NAME,
        "model_version": extractor.MODEL_VERSION,
        "vector_dim": extractor.VECTOR_DIM,
        "endpoints": {
            "health": "/health",
            "extract_file": "/api/v1/extract-features (POST multipart/form-data)",
            "extract_base64": "/api/v1/extract-features/base64 (POST json)"
        }
    }

@app.get("/health")
def health():
    is_ready = getattr(extractor, "is_ready", False)
    return {
        "status": "healthy" if is_ready else "degraded",
        "model": extractor.MODEL_NAME,
        "model_version": extractor.MODEL_VERSION,
        "runtime": "onnxruntime",
        "vector_dim": extractor.VECTOR_DIM,
        "is_ready": is_ready,
        "use_jit": is_ready,
    }

@app.post("/api/v1/extract-features", response_model=ExtractResponse)
def extract_features_upload(file: UploadFile = File(...)):
    """
    Mengekstrak vektor embedding 576-dimensi dari file gambar (JPG/PNG/WEBP)
    """
    try:
        start_time = time.perf_counter()
        contents = file.file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File gambar kosong"
            )

        # Ekstraksi vektor fitur menggunakan extractor MobileNetV3
        vec = extractor.extract_from_bytes(contents)
        inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ExtractResponse(
            success=True,
            dimension=int(len(vec)),
            inference_ms=inference_ms,
            embedding=vec.astype(float).tolist()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses gambar: {str(e)}"
        )

@app.post("/api/v1/extract-features/base64", response_model=ExtractResponse)
def extract_features_base64(req: Base64ImageRequest):
    """
    Mengekstrak vektor embedding 576-dimensi dari Base64 string gambar (data:image/jpeg;base64,...)
    """
    try:
        start_time = time.perf_counter()
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)
        vec = extractor.extract_from_bytes(image_bytes)
        inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ExtractResponse(
            success=True,
            dimension=int(len(vec)),
            inference_ms=inference_ms,
            embedding=vec.astype(float).tolist()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses gambar base64: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
