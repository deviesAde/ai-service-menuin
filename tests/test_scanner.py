import os
import sys
import numpy as np
from PIL import Image

# Setup sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import init_db, add_product, get_products_with_embeddings, delete_product
from backend.feature_extractor import FeatureExtractor

def test_feature_extractor():
    """Menguji ekstraksi fitur MobileNetV3 menghasilkan 576 dimensi L2-normalized"""
    extractor = FeatureExtractor()
    test_img = Image.new("RGB", (300, 300), color=(120, 60, 200))
    vec = extractor.extract(test_img)
    
    assert isinstance(vec, np.ndarray), "Output harus berupa numpy array"
    assert vec.shape == (576,), f"Dimensi vektor harus 576, bukan {vec.shape}"
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3), "Vektor harus ter-normalisasi L2"

def test_database_crud():
    """Menguji operasi SQLite (tambah, ambil, hapus)"""
    init_db()
    
    # Tambah produk uji
    dummy_vec = np.random.randn(576).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    
    prod_id = add_product(
        name="Produk Uji CI",
        price=15000,
        stock=10,
        category="Test",
        image_url="uploads/test.jpg",
        embedding=dummy_vec
    )
    assert prod_id > 0, "ID produk harus bernilai positif"
    
    # Ambil produk & matriks
    prods, matrix = get_products_with_embeddings(use_cache=False)
    assert any(p["id"] == prod_id for p in prods), "Produk harus ditemukan di database"
    assert matrix.shape[1] == 576, "Dimensi matriks embedding harus 576"
    
    # Hapus produk uji
    success = delete_product(prod_id)
    assert success is True, "Produk harus berhasil dihapus"

def test_similarity_math():
    """Menguji perhitungan Cosine Similarity identik menghasilkan skor 1.0"""
    v1 = np.random.randn(576).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    
    # Self similarity
    score = np.dot(v1, v1)
    assert np.isclose(score, 1.0, atol=1e-4), "Skor kemiripan gambar identik harus bernilai 1.0"

if __name__ == "__main__":
    print("Menjalankan Unit Tests...")
    test_feature_extractor()
    print("✅ test_feature_extractor PASSED")
    test_database_crud()
    print("✅ test_database_crud PASSED")
    test_similarity_math()
    print("✅ test_similarity_math PASSED")
    print("\n🎉 SEMUA TES BERHASIL 100%!")

