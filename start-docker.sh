#!/bin/bash
echo "========================================================"
echo "  ScanToko AI - Menjalankan dengan Docker"
echo "========================================================"
echo "Membangun dan menyalakan container Docker..."

docker compose up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "  Sukses! Aplikasi berjalan di: http://localhost:8501"
    echo "========================================================"
    # Coba buka browser otomatis jika tersedia
    if which xdg-open > /dev/null; then
        xdg-open http://localhost:8501
    elif which open > /dev/null; then
        open http://localhost:8501
    fi
else
    echo "[ERROR] Gagal menyalakan container. Pastikan Docker service sudah menyala."
fi
