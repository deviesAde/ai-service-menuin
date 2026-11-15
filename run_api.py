import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Menuin AI Visual Scanner Service (FastAPI)")
    print("=" * 60)
    print("Menjalankan server FastAPI Uvicorn...")
    print("• Base URL  : http://localhost:8000")
    print("• Docs/Swagger : http://localhost:8000/docs")
    print("• Health Check : http://localhost:8000/health")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    subprocess.run(cmd)
