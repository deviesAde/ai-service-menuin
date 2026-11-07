import sqlite3
import numpy as np
import os
import threading
import json
import base64
from typing import List, Dict, Any, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products.db")

# Thread lock & In-memory cache for 0ms retrieval
_cache_lock = threading.Lock()
_cached_products: List[Dict[str, Any]] = None
_cached_matrix: np.ndarray = None

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    # Try optimal WAL mode, with fallback for restricted cloud filesystems
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode = DELETE;")
        
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        price INTEGER NOT NULL,
        stock INTEGER NOT NULL,
        category TEXT DEFAULT 'Kemasan',
        image_url TEXT,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()
    invalidate_cache()

def invalidate_cache():
    global _cached_products, _cached_matrix
    with _cache_lock:
        _cached_products = None
        _cached_matrix = None

def add_product(name: str, price: int, stock: int, category: str, image_url: str, embedding: np.ndarray) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Ensure vector is strictly L2-normalized float32
    embedding = embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm
        
    embedding_blob = embedding.tobytes()
    
    cursor.execute("""
        INSERT INTO products (name, price, stock, category, image_url, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, price, stock, category, image_url, embedding_blob))
    
    product_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    invalidate_cache()
    return product_id

def get_all_products() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, stock, category, image_url, created_at FROM products ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_products_with_embeddings(use_cache: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Returns (products_list, stacked_embeddings_matrix)
    If use_cache is True, retrieves instantly from RAM (0.001 ms) without touching disk!
    """
    global _cached_products, _cached_matrix
    
    if use_cache and _cached_products is not None and _cached_matrix is not None:
        return _cached_products, _cached_matrix

    with _cache_lock:
        if use_cache and _cached_products is not None and _cached_matrix is not None:
            return _cached_products, _cached_matrix
            
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, price, stock, category, image_url, embedding FROM products")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            _cached_products = []
            _cached_matrix = np.empty((0, 576), dtype=np.float32)
            return _cached_products, _cached_matrix
            
        products = []
        embeddings = []
        
        for row in rows:
            p_dict = {
                "id": row["id"],
                "name": row["name"],
                "price": row["price"],
                "stock": row["stock"],
                "category": row["category"],
                "image_url": row["image_url"]
            }
            products.append(p_dict)
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            # Re-verify unit norm
            v_norm = np.linalg.norm(vec)
            if v_norm > 1e-6:
                vec = vec / v_norm
            embeddings.append(vec)
            
        _cached_products = products
        _cached_matrix = np.vstack(embeddings)
        return _cached_products, _cached_matrix

def delete_product(product_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    invalidate_cache()
    return deleted

def export_catalog_json() -> str:
    """Exports all products & embeddings to JSON string for backup/restore"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, stock, category, image_url, embedding FROM products")
    rows = cursor.fetchall()
    conn.close()
    
    items = []
    for r in rows:
        items.append({
            "name": r["name"],
            "price": r["price"],
            "stock": r["stock"],
            "category": r["category"],
            "image_url": r["image_url"],
            "embedding_b64": base64.b64encode(r["embedding"]).decode("utf-8")
        })
    return json.dumps(items, indent=2)

def import_catalog_json(json_str: str) -> int:
    """Restores products from JSON backup string"""
    data = json.loads(json_str)
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    for item in data:
        emb_bytes = base64.b64decode(item["embedding_b64"])
        cursor.execute("""
            INSERT INTO products (name, price, stock, category, image_url, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item["name"], item["price"], item["stock"], item["category"], item.get("image_url", ""), emb_bytes))
        count += 1
    conn.commit()
    conn.close()
    invalidate_cache()
    return count
