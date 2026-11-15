@echo off
echo ========================================================
echo   ScanToko AI - Menjalankan dengan Docker
echo ========================================================
echo Membangun dan menyalakan container Docker...
echo Tidak perlu install Python atau PyTorch!
echo.

echo Mengunduh atau menyiapkan container...
docker compose pull
docker compose up -d --build

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pastikan aplikasi Docker Desktop sudah menyala di laptop Anda.
    pause
    exit /b
)

echo.
echo ========================================================
echo   Sukses! Aplikasi telah berjalan.
echo   Membuka browser di http://localhost:8501 ...
echo ========================================================
start http://localhost:8501
pause
