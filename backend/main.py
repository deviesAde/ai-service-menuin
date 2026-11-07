import os
import sys
import time
import uuid
import numpy as np

# Pastikan folder backend ada di sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import (
    init_db,
    add_product,
    get_all_products,
    get_products_with_embeddings,
    delete_product
)
from feature_extractor import FeatureExtractor

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="AI Retail Scanner API", version="1.0.0")

# CORS agar bisa diakses dari berbagai origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Inisialisasi DB & AI Extractor saat server start
init_db()
extractor = FeatureExtractor()

# Mount Static Files (Uploads & Frontend)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ==========================================
# ENDPOINT: SCAN PRODUK DARI KAMERA
# ==========================================
@app.post("/api/scan")
async def scan_product(file: UploadFile = File(...)):
    """
    Menerima snapshot kemasan produk (224x224 WebP/JPEG),
    mengekstrak vektor 512-dim, dan mencocokkan dengan ratusan produk di SQLite (< 1ms).
    """
    total_start = time.perf_counter()

    # 1. Baca bytes gambar dari request
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File gambar kosong")

    # 2. Ekstrak Vektor Fitur (512 dimensi)
    ext_start = time.perf_counter()
    query_vector = extractor.extract_from_bytes(image_bytes)
    extraction_ms = (time.perf_counter() - ext_start) * 1000

    # 3. Ambil data produk & matrix embedding dari RAM Cache (0 ms disk I/O)
    search_start = time.perf_counter()
    products, embeddings_matrix = get_products_with_embeddings(use_cache=True)

    if len(products) == 0:
        return {
            "found": False,
            "message": "Katalog toko masih kosong. Silakan tambahkan produk terlebih dahulu.",
            "latency": {"extraction_ms": round(extraction_ms, 2), "total_ms": round(extraction_ms, 2)}
        }

    # 4. Vector Search: Hitung Cosine Similarity ke seluruh katalog produk secara paralel
    # Dot product karena kedua vektor sudah ter-normalisasi L2 (magnitude = 1)
    similarities = np.dot(embeddings_matrix, query_vector)
    search_ms = (time.perf_counter() - search_start) * 1000

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    matched_product = products[best_idx]

    total_ms = (time.perf_counter() - total_start) * 1000

    # Threshold kecocokan visual (misal > 0.65 dianggap cocok)
    MATCH_THRESHOLD = 0.65
    is_match = best_score >= MATCH_THRESHOLD

    return {
        "found": is_match,
        "product": matched_product if is_match else None,
        "best_candidate": matched_product if not is_match else None,
        "similarity": round(best_score, 4),
        "confidence_percent": round(max(0.0, min(100.0, best_score * 100)), 1),
        "threshold": MATCH_THRESHOLD,
        "latency": {
            "extraction_ms": round(extraction_ms, 2),
            "search_ms": round(search_ms, 2),
            "total_ms": round(total_ms, 2)
        },
        "total_products_checked": len(products)
    }


# ==========================================
# ENDPOINT: REGISTER / TAMBAH PRODUK BARU
# ==========================================
@app.post("/api/products")
async def create_product(
    name: str = Form(...),
    price: int = Form(...),
    stock: int = Form(...),
    category: str = Form("Kemasan"),
    file: UploadFile = File(...)
):
    """
    Daftarkan produk baru: upload foto kemasan -> ekstrak vektor -> simpan ke SQLite.
    Instan siap scan tanpa perlu retrain model!
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Foto kemasan wajib diunggah")

    # Simpan file gambar kemasan
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"prod_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # Ekstrak vektor embedding 512 dimensi
    embedding = extractor.extract_from_bytes(image_bytes)

    # Simpan ke SQLite
    image_url = f"/uploads/{filename}"
    prod_id = add_product(name, price, stock, category, image_url, embedding)

    return {
        "success": True,
        "message": "Produk berhasil didaftarkan dan langsung siap dipindai!",
        "product_id": prod_id,
        "name": name,
        "image_url": image_url
    }


# ==========================================
# ENDPOINT: LIHAT SEMUA PRODUK
# ==========================================
@app.get("/api/products")
async def list_products():
    return get_all_products()


# ==========================================
# ENDPOINT: HAPUS PRODUK
# ==========================================
@app.delete("/api/products/{product_id}")
async def remove_product(product_id: int):
    success = delete_product(product_id)
    if not success:
        raise HTTPException(status_code=404, detail="Produk tidak ditemukan")
    return {"success": True, "message": "Produk berhasil dihapus"}


# ==========================================
# SERVE FRONTEND WEB
# ==========================================
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount aset statis frontend lainnya (js, css)
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
