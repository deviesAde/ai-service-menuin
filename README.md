# ScanToko AI - Scanner Produk Kemasan Toko

Sistem scanner produk kemasan toko menggunakan antarmuka Streamlit, model visi MobileNetV3 (TorchScript JIT), dan pencarian kemiripan vektor SQLite (Cosine Similarity) dengan latensi pemrosesan di bawah 10 milidetik.

---

## Pilihan Cara Menjalankan Aplikasi

Terdapat beberapa metode untuk menjalankan sistem ini, silakan pilih salah satu sesuai kebutuhan lingkungan Anda.

### Pilihan 1: Menggunakan Docker Image Siap Pakai (Rekomendasi Tanpa Install)

Metode ini paling praktis untuk pengguna atau rekan tim yang tidak ingin menginstal Python, PyTorch, maupun pustaka lainnya. Cukup pastikan Docker Desktop telah berjalan di komputer.

Jalankan perintah berikut di terminal:

```bash
docker run -d -p 8501:8501 --name scantoko-ai ghcr.io/deviesade/ai-service-menuin:latest
```

Buka peramban (browser) dan akses alamat:
`http://localhost:8501`

Untuk menghentikan container:
```bash
docker stop scantoko-ai
```

---

### Pilihan 2: Menggunakan Docker Compose (Data Tersimpan di Komputer Lokal)

Metode ini disarankan jika Anda menjalankan aplikasi di dalam container namun ingin file gambar yang diunggah dan database SQLite tetap tersimpan di folder lokal komputer.

1. Buka folder proyek ini di terminal.
2. Jalankan perintah berikut:
   ```bash
   docker compose up -d
   ```
   *(Untuk pengguna Windows, Anda juga dapat mengklik ganda file `start-docker.bat`)*.
3. Akses aplikasi melalui alamat `http://localhost:8501`.
4. Untuk mematikan container:
   ```bash
   docker compose down
   ```

---

### Pilihan 3: Menjalankan Langsung Menggunakan Python

Metode ini cocok untuk pengembangan lokal jika komputer Anda sudah memiliki Python 3.10 atau 3.11.

1. Pasang seluruh dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi Streamlit:
   ```bash
   python run.py
   ```
   atau:
   ```bash
   streamlit run app.py
   ```
3. Buka peramban pada alamat `http://localhost:8501`.

---

### Pilihan 4: Menjalankan Mode FastAPI (Microservice Backend)

Metode ini digunakan jika aplikasi ingin dihubungkan sebagai backend REST API untuk integrasi sistem Web POS lain:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi API interaktif Swagger dapat diakses di:
`http://localhost:8000/docs`

---

## Langkah-Langkah Penggunaan Aplikasi

Setelah aplikasi berhasil terbuka di browser, ikuti tahapan berikut:

### Langkah 1: Mendaftarkan Produk Baru
1. Buka tab "Tambah / Upload Produk Baru".
2. Masukkan nama produk, kategori, harga satuan, dan jumlah stok awal.
3. Pilih metode foto: unggah file gambar kemasan atau gunakan kamera laptop/HP secara langsung.
4. Klik tombol "Daftarkan Produk ke Database AI".
5. Sistem akan mengekstrak vektor karakteristik visual produk dan menyimpannya ke database SQLite. Produk langsung dapat dipindai seketika tanpa perlu proses pelatihan ulang (zero-shot).

### Langkah 2: Memindai Produk Toko
1. Buka tab "Scan Produk".
2. Pilih salah satu dari tiga mode scanner yang tersedia:
   * Live Real-Time (Video 30 FPS): Kamera streaming berjalan terus-menerus. Arahkan produk ke kotak bidik di layar, dan hasil nama serta harga produk akan langsung muncul otomatis di layar.
   * Snapshot Kamera: Mengambil foto produk satu per satu secara manual.
   * Upload File Foto: Menguji file gambar dari penyimpanan komputer atau galeri HP.
3. Hasil identifikasi produk, harga satuan, sisa stok, dan persentase kemiripan akan ditampilkan secara instan.

### Langkah 3: Pencadangan dan Pemulihan Data (Backup & Restore)
1. Buka tab "Katalog Produk (SQLite)".
2. Klik tombol "Download Backup Database (JSON)" untuk mengunduh seluruh data produk beserta embedding visualnya dalam satu file cadangan.
3. Gunakan menu "Restore Database dari File Backup (JSON)" jika ingin memindahkan atau memulihkan data produk ke server atau perangkat lain.

---

## Otomatisasi CI/CD dan Pembuatan Docker Image

Repository ini telah dilengkapi dengan pipeline GitHub Actions otomatis:

1. CI Pipeline (`.github/workflows/ci.yml`):
   Setiap kali ada pembaruan kode yang di-push ke branch utama, GitHub akan otomatis menjalankan pengecekan sintaks dan pengujian unit (`tests/test_scanner.py`).

2. Docker Registry Publisher (`.github/workflows/docker-publish.yml`):
   GitHub Actions akan otomatis membangun Docker image dan mempublikasikannya ke GitHub Container Registry (`ghcr.io/deviesade/ai-service-menuin:latest`).

---

## Struktur Direktori Proyek

```
ai-service-menuin/
|-- .github/
|   `-- workflows/
|       |-- ci.yml                  # Konfigurasi Continuous Integration
|       `-- docker-publish.yml      # Otomatisasi build & publish image Docker
|-- backend/
|   |-- __init__.py                 # Marker paket Python backend
|   |-- database.py                 # Manajemen SQLite, WAL mode, dan in-memory cache
|   |-- feature_extractor.py        # Model MobileNetV3 TorchScript JIT
|   |-- realtime_processor.py       # Pemroses video real-time 30 FPS dan AR HUD
|   |-- main.py                     # Endpoint API FastAPI
|   `-- products.db                 # Berkas database SQLite lokal
|-- tests/
|   `-- test_scanner.py             # Kumpulan pengujian unit otomatis
|-- uploads/                        # Direktori penyimpanan foto produk
|-- app.py                          # Aplikasi antarmuka utama Streamlit
|-- run.py                          # Skrip peluncur Streamlit
|-- Dockerfile                      # Definisi container image Docker
|-- docker-compose.yml              # Konfigurasi container service dan volume
|-- start-docker.bat                # Skrip peluncur Docker untuk Windows
|-- start-docker.sh                 # Skrip peluncur Docker untuk Mac/Linux
|-- requirements.txt                # Daftar pustaka Python
`-- README.md                       # Dokumentasi resmi proyek
```
