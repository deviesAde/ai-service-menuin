import os
import sys
import time
import uuid
import numpy as np
from PIL import Image
import streamlit as st

# Setup sys.path agar modul backend dapat diakses di cloud maupun lokal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
backend_dir = os.path.join(BASE_DIR, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from backend.database import (
        init_db,
        add_product,
        get_all_products,
        get_products_with_embeddings,
        delete_product,
        export_catalog_json,
        import_catalog_json
    )
    from backend.feature_extractor import FeatureExtractor
except (ImportError, ModuleNotFoundError):
    from database import (
        init_db,
        add_product,
        get_all_products,
        get_products_with_embeddings,
        delete_product,
        export_catalog_json,
        import_catalog_json
    )
    from feature_extractor import FeatureExtractor

# ==========================================
# STREAMLIT CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="ScanToko AI - Scanner Produk Toko",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: 800;
        color: #10b981;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Database & Direktori Uploads
init_db()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cache model MobileNetV3 di RAM agar selalu hangat & cepat (~5ms)
@st.cache_resource
def load_feature_extractor():
    return FeatureExtractor()

extractor = load_feature_extractor()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚡ ScanToko AI")
    st.caption("Scanner Produk Toko • MobileNetV3 • SQLite Fast Cache")
    st.write("")
    
    products, _ = get_products_with_embeddings(use_cache=True)
    st.metric("Total Produk Terdaftar", f"{len(products)} SKU")
    
    st.divider()
    st.markdown("### ⚙️ Pengaturan Sensitivitas")
    threshold = st.slider(
        "Batas Kemiripan (Threshold)",
        min_value=40,
        max_value=90,
        value=60,
        step=5,
        help="Semakin rendah nilai threshold, semakin mudah produk terdeteksi meski pencahayaan/sudut kamera berbeda."
    ) / 100.0

    st.divider()
    if len(products) > 0:
        if st.button("🗑️ Kosongkan Database (Hapus Semua)", type="secondary"):
            for p in products:
                delete_product(p["id"])
            st.toast("Seluruh produk berhasil dihapus!")
            st.rerun()

    st.markdown("---")
    st.caption("v2.2 • Cloud & Mobile Ready")


# ==========================================
# MAIN TABS
# ==========================================
tab_scan, tab_upload, tab_catalog, tab_bench = st.tabs([
    "📷 Scan Produk",
    "➕ Tambah / Upload Produk Baru",
    "📦 Katalog Produk (SQLite)",
    "⚡ Arsitektur & Performa"
])


# ==============================================================================
# TAB 1: SCAN PRODUK
# ==============================================================================
with tab_scan:
    st.subheader("📷 Pemindaian Produk Toko")
    
    products, embeddings_matrix = get_products_with_embeddings(use_cache=True)
    
    if len(products) == 0:
        st.info("💡 **Database toko Anda masih kosong.** Silakan masuk ke tab **'➕ Tambah / Upload Produk Baru'** untuk mendaftarkan produk toko Anda terlebih dahulu!")
    else:
        st.write(f"Siap mendeteksi produk dari **{len(products)} produk** terdaftar.")

        scan_method = st.radio(
            "Pilih Metode Pemindaian:",
            ["📸 Kamera Langsung (Snapshot Instan)", "📁 Upload Foto dari Komputer/HP"],
            horizontal=True
        )

        query_image = None

        if scan_method == "📸 Kamera Langsung (Snapshot Instan)":
            st.caption("Arahkan kamera ke kemasan produk, lalu klik tombol jepret di bawah:")
            cam_file = st.camera_input("Kamera Scanner")
            if cam_file is not None:
                query_image = Image.open(cam_file)
        else:
            uploaded_file = st.file_uploader("Upload foto kemasan produk yang ingin dipindai:", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                query_image = Image.open(uploaded_file)

        # ================= EKSEKUSI PENCOCOKAN AI =================
        if query_image is not None:
            st.divider()
            col_img, col_res = st.columns([1, 1.3])
            
            with col_img:
                st.image(query_image, caption="Foto yang Sedang Dipindai", use_container_width=True)

            with col_res:
                total_start = time.perf_counter()

                # 1. Ekstraksi Vektor Fitur (TorchScript JIT ~5ms)
                ext_start = time.perf_counter()
                query_vec = extractor.extract(query_image)
                ext_ms = (time.perf_counter() - ext_start) * 1000

                # 2. Vector Search: Cosine Similarity (< 0.05 ms)
                search_start = time.perf_counter()
                similarities = np.dot(embeddings_matrix, query_vec)
                search_ms = (time.perf_counter() - search_start) * 1000

                # Urutkan dari skor tertinggi
                ranked_indices = np.argsort(similarities)[::-1]
                best_idx = int(ranked_indices[0])
                best_score = float(similarities[best_idx])
                best_product = products[best_idx]
                total_ms = (time.perf_counter() - total_start) * 1000

                # 3. Tampilkan Metrik Latensi
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⚡ Total Waktu", f"{total_ms:.1f} ms")
                m2.metric("🧠 Ekstraksi AI", f"{ext_ms:.1f} ms")
                m3.metric("🔍 Vektor Search", f"{search_ms:.3f} ms")
                m4.metric("🎯 Akurasi", f"{best_score * 100:.1f} %")

                st.write("")

                # 4. Keputusan Pencocokan
                if best_score >= threshold:
                    st.success("✅ **PRODUK BERHASIL DIKENALI!**")
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            img_path = os.path.join(BASE_DIR, best_product["image_url"]) if best_product["image_url"] else ""
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.write("🖼️ [Foto Database]")
                        with c2:
                            st.markdown(f"### {best_product['name']}")
                            st.caption(f"Kategori: **{best_product['category']}** • ID: #{best_product['id']}")
                            st.markdown(f"<div class='price-tag'>Rp {best_product['price']:,}</div>".replace(",", "."), unsafe_allow_html=True)
                            st.markdown(f"📦 Stok Tersedia: **{best_product['stock']} pcs**")
                            st.progress(min(max(float(best_score), 0.0), 1.0), text=f"Tingkat Kemiripan Visual: {best_score*100:.1f}%")
                
                elif best_score >= 0.45:
                    st.warning("⚠️ **KEMUNGKINAN COCOK (Kemiripan Menengah)**")
                    st.write(f"Kandidat terdekat: **{best_product['name']}** ({best_score*100:.1f}%)")
                    st.info("💡 Pastikan produk dihadapkan tegak, lebih dekat ke lensa, atau turunkan sedikit slider batas kemiripan di sidebar.")
                    with st.container(border=True):
                        st.markdown(f"**{best_product['name']}** • Rp {best_product['price']:,}".replace(",", "."))
                        st.progress(float(best_score), text=f"Kemiripan: {best_score*100:.1f}%")
                else:
                    st.error("❌ **PRODUK TIDAK DIKENALI**")
                    st.write("Kemasan produk ini belum terdaftar di database toko Anda.")

                # Tampilkan daftar kandidat jika ada lebih dari 1 produk
                if len(products) > 1:
                    with st.expander("📊 Lihat 3 Peringkat Produk Terdekat"):
                        for i in range(min(3, len(products))):
                            idx = int(ranked_indices[i])
                            sc = float(similarities[idx])
                            prod = products[idx]
                            st.write(f"**#{i+1} {prod['name']}** — Kemiripan: `{sc*100:.1f}%` (Rp {prod['price']:,})".replace(",", "."))


# ==============================================================================
# TAB 2: TAMBAH / UPLOAD PRODUK BARU (FIX: TANPA FORM BUG)
# ==============================================================================
with tab_upload:
    st.subheader("➕ Daftarkan Produk Toko Anda")
    st.write("Unggah foto kemasan asli atau foto langsung lewat kamera. Produk langsung siap di-scan seketika tanpa perlu training ulang!")

    # Notifikasi jika baru saja berhasil menambahkan produk
    if "just_added" in st.session_state:
        st.success(f"🎉 **Sukses!** Produk **{st.session_state['just_added']}** berhasil didaftarkan dan langsung tersimpan di database!")
        del st.session_state["just_added"]

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        nama_produk = st.text_input("Nama Produk Toko:", placeholder="Misal: Indomie Goreng Spesial 85g")
        kategori = st.selectbox(
            "Kategori Produk:",
            ["Makanan Ringan / Snack", "Mi Instan", "Minuman", "Kopi & Teh", "Air Mineral", "Biskuit & Roti", "Sabun & Perawatan", "Bumbu & Sembako", "Lainnya"]
        )
    with col_f2:
        harga = st.number_input("Harga Satuan (Rp):", min_value=100, max_value=100000000, value=3500, step=500)
        stok = st.number_input("Jumlah Stok Awal:", min_value=0, max_value=100000, value=24, step=1)

    st.markdown("#### Foto Kemasan Produk:")
    metode_upload = st.radio("Pilih Cara Ambil Foto:", ["📁 Upload File Foto (Galeri / Komputer)", "📸 Ambil Foto Langsung dengan Kamera"], horizontal=True)

    foto_file = None
    if metode_upload == "📁 Upload File Foto (Galeri / Komputer)":
        foto_file = st.file_uploader("Pilih file foto produk kemasan (JPG / PNG / WebP):", type=["jpg", "jpeg", "png", "webp"])
    else:
        st.caption("Posisikan kemasan produk di depan kamera lalu klik Take Photo:")
        foto_file = st.camera_input("Ambil Foto Produk Baru")

    if foto_file is not None:
        try:
            preview_img = Image.open(foto_file)
            st.image(preview_img, caption="Pratinjau Foto Produk", width=200)
        except Exception:
            pass

    st.write("")
    btn_submit = st.button("💾 Daftarkan Produk ke Database AI", type="primary", use_container_width=True)

    if btn_submit:
        if not nama_produk.strip():
            st.error("Nama produk tidak boleh kosong!")
        elif foto_file is None:
            st.error("Silakan unggah atau ambil foto kemasan produk terlebih dahulu!")
        else:
            with st.spinner("Mengekstrak vektor AI MobileNetV3 & Menyimpan ke SQLite..."):
                try:
                    # 1. Buka dan konversi gambar ke RGB
                    img = Image.open(foto_file).convert("RGB")
                    
                    # 2. Simpan fisik file gambar sebagai standar JPEG
                    unique_filename = f"prod_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(UPLOAD_DIR, unique_filename)
                    img.save(save_path, "JPEG", quality=90)
                    
                    # 3. Ekstrak Vektor Embedding (576 dimensi L2-normalized)
                    embedding = extractor.extract(img)
                    
                    # 4. Simpan ke database SQLite
                    rel_url = f"uploads/{unique_filename}"
                    new_id = add_product(
                        name=nama_produk.strip(),
                        price=int(harga),
                        stock=int(stok),
                        category=kategori,
                        image_url=rel_url,
                        embedding=embedding
                    )
                    
                    # Catat ke session state dan reload agar UI langsung terupdate
                    st.session_state["just_added"] = nama_produk.strip()
                    st.rerun()

                except Exception as e:
                    st.error(f"Gagal menyimpan produk: {e}")


# ==============================================================================
# TAB 3: KATALOG PRODUK DI SQLITE & BACKUP RESTORE
# ==============================================================================
with tab_catalog:
    st.subheader("📦 Katalog Produk di Database SQLite")
    all_prods = get_all_products()
    st.write(f"Saat ini terdapat **{len(all_prods)} produk** tersimpan di database.")

    # Fitur Backup & Restore
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if len(all_prods) > 0:
            json_backup = export_catalog_json()
            st.download_button(
                label="📥 Download Backup Database (JSON)",
                data=json_backup,
                file_name="backup_katalog_toko.json",
                mime="application/json",
                help="Unduh file cadangan katalog Anda untuk disimpan atau dipindahkan ke server lain."
            )
    with col_b2:
        restore_file = st.file_uploader("📤 Restore Database dari File Backup (JSON):", type=["json"], key="restore_uploader")
        if restore_file is not None:
            if st.button("Terapkan Restore Database", type="secondary"):
                content = restore_file.read().decode("utf-8")
                restored = import_catalog_json(content)
                st.success(f"Berhasil merestore {restored} produk ke database!")
                st.rerun()

    st.divider()

    if not all_prods:
        st.info("Belum ada produk yang didaftarkan. Gunakan tab **'➕ Tambah / Upload Produk Baru'** untuk menambahkan produk toko Anda.")
    else:
        cols = st.columns(3)
        for idx, p in enumerate(all_prods):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    img_disk = os.path.join(BASE_DIR, p["image_url"]) if p["image_url"] else ""
                    if img_disk and os.path.exists(img_disk):
                        st.image(img_disk, use_container_width=True)
                    else:
                        st.caption("🖼️ [Tidak ada foto]")
                    
                    st.markdown(f"**{p['name']}**")
                    st.caption(f"ID: #{p['id']} • {p['category']}")
                    st.markdown(f"💰 **Rp {p['price']:,}** &nbsp;&nbsp;|&nbsp;&nbsp; 📦 **{p['stock']} pcs**".replace(",", "."))
                    
                    if st.button("🗑️ Hapus", key=f"del_{p['id']}", use_container_width=True):
                        delete_product(p["id"])
                        st.toast(f"Produk #{p['id']} berhasil dihapus!")
                        st.rerun()


# ==============================================================================
# TAB 4: ARSITEKTUR & BENCHMARK
# ==============================================================================
with tab_bench:
    st.subheader("⚡ Arsitektur & Analisis Kecepatan Turbo")
    
    st.markdown("""
    Sistem ini dirancang untuk **kecepatan maksimal (< 10 ms)** dan **kompatibilitas 100% pada Cloud & Mobile**:
    
    1. **MobileNetV3 Small (TorchScript JIT):**
       * Model deep learning terkompilasi mesin C++ yang mengekstrak 576 dimensi fitur dalam **~5 ms**.
    2. **Preservasi Visual Utuh:**
       * Tidak memotong gambar secara agresif sehingga seluruh kemasan, logo, teks, dan proporsi produk dikenali akurat.
    3. **In-Memory Zero-Disk Cache:**
       * Matriks vektor produk berada langsung di RAM sehingga pencarian kemiripan Cosine Similarity berjalan dalam **< 0.05 ms**.
    4. **Fitur Backup & Restore:**
       * Mendukung ekspor-impor database JSON sehingga katalog toko Anda aman dan tidak hilang saat server di-deploy ulang.
    """)

    st.divider()
    st.markdown("### 🧪 Uji Benchmark Kecepatan Nyata")
    if st.button("🚀 Jalankan Benchmark (100 Iterasi)", type="primary"):
        with st.spinner("Menguji kecepatan pipeline..."):
            sample_img = Image.new("RGB", (720, 720), color=(100, 150, 200))
            db_sim = np.random.randn(1000, 576).astype(np.float32)
            db_sim /= np.linalg.norm(db_sim, axis=1, keepdims=True)

            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                v = extractor.extract(sample_img)
                _ = np.dot(db_sim, v)
                times.append((time.perf_counter() - t0) * 1000)

            avg_t = np.mean(times)
            min_t = np.min(times)

            st.success(f"🎉 Rata-rata waktu per frame: **{avg_t:.2f} ms**!")
            b1, b2 = st.columns(2)
            b1.metric("⚡ Rata-rata", f"{avg_t:.2f} ms")
            b2.metric("🚀 Tercepat", f"{min_t:.2f} ms")
