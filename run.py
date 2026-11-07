import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 60)
    print("⚡ ScanToko AI - Streamlit Turbo Edition")
    print("=" * 60)
    print("Memulai aplikasi Streamlit...")
    print("• Akses di Laptop/PC : http://localhost:8501")
    print("• Akses di HP (Wi-Fi): http://192.168.1.8:8501")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=false"
    ]
    subprocess.run(cmd)
