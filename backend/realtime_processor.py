import cv2
import numpy as np
import time
from typing import Tuple, Optional
from database import get_products_with_embeddings
from feature_extractor import FeatureExtractor

try:
    import av
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    # Fallback dummy class if av/webrtc not available
    class VideoProcessorBase:
        pass

class RealtimeScannerProcessor(VideoProcessorBase):
    """
    Processor Real-Time 30 FPS untuk streamlit-webrtc:
    Menerima setiap frame kamera video, memotong area tengah,
    mencocokkan dengan katalog produk di RAM (< 5ms),
    dan menggambar HUD AR interaktif langsung di atas video streaming.
    """
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold
        self.extractor = FeatureExtractor()
        self.last_match = None
        self.last_score = 0.0
        self.last_latency = 0.0
        self.frame_count = 0
        self.process_every_n_frames = 2 # Proses tiap 2 frame untuk efisiensi CPU maksimal (~15-20 FPS inference)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def recv(self, frame: 'av.VideoFrame') -> 'av.VideoFrame':
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        self.frame_count += 1

        # 1. Tentukan Kotak Bidik Tengah (Center Target Box)
        box_size = int(min(h, w) * 0.65)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # 2. Proses AI (tiap n frame atau gunakan hasil frame sebelumnya)
        if self.frame_count % self.process_every_n_frames == 0:
            t0 = time.perf_counter()
            
            # Potong bagian dalam kotak bidik
            crop_bgr = img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            
            # Ekstraksi Vektor Fitur (TorchScript JIT ~5ms)
            query_vec = self.extractor._extract_from_numpy(crop_rgb)
            
            # Vector Search di RAM
            products, matrix = get_products_with_embeddings(use_cache=True)
            
            if len(products) > 0 and matrix.shape[0] > 0:
                sims = np.dot(matrix, query_vec)
                best_idx = int(np.argmax(sims))
                self.last_score = float(sims[best_idx])
                if self.last_score >= self.threshold:
                    self.last_match = products[best_idx]
                else:
                    self.last_match = None
            else:
                self.last_match = None
                self.last_score = 0.0
                
            self.last_latency = (time.perf_counter() - t0) * 1000

        # 3. Gambar HUD Interaktif (Overlay Augmented Reality)
        is_detected = self.last_match is not None and self.last_score >= self.threshold
        accent_color = (0, 255, 120) if is_detected else (0, 220, 255) # Hijau neon atau Cyan

        # Gambar Sudut Kotak Bidik (Corner Brackets)
        c_len = int(box_size * 0.12)
        th = 3
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + c_len, y1), accent_color, th)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), accent_color, th)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - c_len, y1), accent_color, th)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), accent_color, th)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + c_len, y2), accent_color, th)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), accent_color, th)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - c_len, y2), accent_color, th)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), accent_color, th)

        # Gambar Laser Scan Animasi
        laser_y = int(y1 + (y2 - y1) * (0.5 + 0.45 * np.sin(self.frame_count * 0.15)))
        cv2.line(img, (x1 + 10, laser_y), (x2 - 10, laser_y), accent_color, 1)

        # 4. Gambar Banner Hasil Deteksi di Bawah Frame
        if is_detected:
            p = self.last_match
            # Semi-transparent dark banner
            overlay = img.copy()
            banner_h = 80
            cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            
            # Border atas banner
            cv2.line(img, (0, h - banner_h), (w, h - banner_h), (0, 255, 120), 2)
            
            # Teks Nama Produk
            cv2.putText(img, f"{p['name']}", (20, h - 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
            
            # Teks Harga & Stok
            price_text = f"Rp {p['price']:,}".replace(",", ".")
            cv2.putText(img, f"{price_text}  |  Stok: {p['stock']} pcs", (20, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 120), 2)
            
            # Akurasi & Latensi di kanan banner
            stat_text = f"{self.last_score*100:.1f}%  ({self.last_latency:.0f}ms)"
            cv2.putText(img, stat_text, (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        else:
            # Banner mencari kemasan
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y2 + 10), (x2, y2 + 40), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            cv2.putText(img, "Arahkan kemasan ke dalam kotak", (x1 + 15, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")
