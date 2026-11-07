# ⚡ ScanToko AI (Streamlit Edition)

Scanner produk kemasan toko mandiri menggunakan **Streamlit**, model Deep Learning **MobileNetV3**, dan pencarian kemiripan vektor **SQLite (Cosine Similarity)** dengan latensi **< 100 ms**.

---

## 🌟 Cara Kerja Sistem

1. **Database Bersih (Siap Diisi Produk Sendiri):**
   * Database SQLite dimulai dalam keadaan kosong (0 SKU).
   * Anda memiliki kendali penuh untuk mendaftarkan produk kemasan toko Anda sendiri.
2. **Tambah / Upload Produk Mandiri:**
   * Di tab **"➕ Tambah / Upload Produk Baru"**, Anda dapat mengunggah file foto kemasan dari galeri komputer/HP atau mengambil foto langsung lewat kamera.
   * Masukkan nama produk, harga (Rp), dan stok.
   * AI MobileNetV3 langsung mengekstrak vektor visual 576-dimensi dan menyimpannya ke SQLite.
   * **Tanpa Retraining (Zero-Shot):** Produk yang baru Anda daftarkan langsung bisa dipindai detik itu juga!
3. **Pindai via Kamera / File:**
   * Di tab **"📷 Scan Produk"**, arahkan kamera ke produk toko Anda atau unggah foto untuk dipindai.
   * Sistem menghitung kemiripan (*Cosine Similarity*) dengan seluruh katalog di SQLite dalam waktu **< 1 milidetik**.

---

## 🚀 Cara Menjalankan

Cukup jalankan perintah berikut di terminal:

```bash
python run.py
```

atau langsung via Streamlit:

```bash
streamlit run app.py
```

Aplikasi otomatis terbuka di browser pada alamat:
👉 **`http://localhost:8501`**

---

## 📁 Struktur File

```
ai-service-menuin/
│
├── app.py                      # Aplikasi Utama Streamlit
├── run.py                      # Script peluncur Streamlit
│
├── backend/
│   ├── database.py             # SQLite helper (tabel products & simpan embedding BLOB)
│   ├── feature_extractor.py    # Ekstraksi vektor 576-dimensi (PyTorch MobileNetV3)
│   └── products.db             # Database SQLite lokal (mulai dari 0 produk)
│
├── uploads/                    # Folder penyimpanan foto produk yang Anda upload (saat ini bersih)
│
└── README.md
```
