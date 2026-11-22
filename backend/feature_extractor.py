import io
import os
import urllib.request
import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Xenova/mobileclip_s0/resolve/main/onnx/vision_model.onnx"
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobileclip_s0_vision.onnx")


class FeatureExtractor:
    """
    Ekstraktor Fitur Visual Berbasis ONNX Runtime (MobileCLIP-S0 Vision Encoder):
    1. Menggunakan MobileCLIP-S0 (512-dimensi, ~50ms CPU, akurasi tinggi untuk logo, teks & kemasan retail).
    2. Preprocessing standar CLIP: resize proporsional (shortest_edge=256) + center-crop 256x256 + rescale [0, 1].
    3. Normalisasi L2 unit norm selalu terjamin.
    """

    MODEL_NAME = "MobileCLIP-S0 (ONNX Runtime)"
    MODEL_VERSION = "mobileclip-s0-onnx-v1"
    VECTOR_DIM = 512
    INPUT_SIZE = 256

    def __init__(self, model_path: str | None = None):
        self.is_ready = False
        self.use_jit = False  # Alias kompatibilitas untuk healthcheck lama
        self.session = None
        self.input_name = "pixel_values"
        self.model_path = model_path or os.environ.get("MOBILECLIP_ONNX_PATH", DEFAULT_MODEL_PATH)
        self.model_url = os.environ.get("MOBILECLIP_ONNX_URL", DEFAULT_MODEL_URL)

        try:
            import onnxruntime as ort

            self._ensure_model_file()

            opts = ort.SessionOptions()
            threads = min(4, os.cpu_count() or 4)
            opts.intra_op_num_threads = threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

            # Warmup 2x
            dummy = np.zeros((1, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=np.float32)
            for _ in range(2):
                _ = self.session.run(None, {self.input_name: dummy})

            self.is_ready = True
            self.use_jit = True
            print(
                f"[AI Engine] {self.MODEL_NAME} aktif ({threads} CPU threads, dim={self.VECTOR_DIM})."
            )
        except Exception as e:
            print(f"[AI Engine] Gagal memuat ONNX MobileCLIP-S0, fallback aktif: {e}")
            self.is_ready = False
            self.use_jit = False

    def _ensure_model_file(self) -> None:
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1_000_000:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"[AI Engine] Mengunduh model ONNX dari {self.model_url}...")
        urllib.request.urlretrieve(self.model_url, self.model_path)

    def extract_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        if cv2 is not None:
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return self._extract_from_numpy(rgb)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.extract(pil_img)

    def extract(self, image: Image.Image) -> np.ndarray:
        rgb_arr = np.array(image.convert("RGB"))
        return self._extract_from_numpy(rgb_arr)

    def _preprocess_clip(self, rgb_arr: np.ndarray) -> np.ndarray:
        """
        Resize proporsional agar sisi terpendek = 256, lalu center-crop 256x256,
        dan rescale [0..255] -> [0.0..1.0] float32 (sesuai preprocessor_config.json MobileCLIP-S0).
        """
        h, w = rgb_arr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Dimensi gambar tidak valid")

        if h != self.INPUT_SIZE or w != self.INPUT_SIZE:
            scale = self.INPUT_SIZE / min(h, w)
            new_w = max(self.INPUT_SIZE, int(round(w * scale)))
            new_h = max(self.INPUT_SIZE, int(round(h * scale)))
            if cv2 is not None:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                resized = cv2.resize(rgb_arr, (new_w, new_h), interpolation=interp)
            else:
                resized = np.array(
                    Image.fromarray(rgb_arr).resize((new_w, new_h), Image.Resampling.BILINEAR)
                )

            y0 = (new_h - self.INPUT_SIZE) // 2
            x0 = (new_w - self.INPUT_SIZE) // 2
            cropped = resized[y0 : y0 + self.INPUT_SIZE, x0 : x0 + self.INPUT_SIZE]
        else:
            cropped = rgb_arr

        chw = (cropped.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0)

    def _extract_from_numpy(self, rgb_arr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess_clip(rgb_arr)

        if self.is_ready and self.session is not None:
            outputs = self.session.run(None, {self.input_name: tensor})
            feat = outputs[0].flatten().astype(np.float32)
            return self._l2_normalize(feat)

        return self._fallback_spatial(tensor[0])

    def _fallback_spatial(self, chw_01: np.ndarray) -> np.ndarray:
        """Fallback 512-dimensi deterministik jika ONNX gagal dimuat."""
        arr = (chw_01.transpose(1, 2, 0) * 255.0).astype(np.float32)
        feats: list[float] = []
        # 4x4 grid x 3 channel x 8 bins = 384
        cell_sz = self.INPUT_SIZE // 4
        for r in range(4):
            for c in range(4):
                cell = arr[r * cell_sz : (r + 1) * cell_sz, c * cell_sz : (c + 1) * cell_sz]
                for ch in range(3):
                    hist, _ = np.histogram(cell[:, :, ch], bins=8, range=(0, 256))
                    feats.extend(hist.tolist())
        # Center region 3 channel x 32 bins + 32 gradient = 128 -> total 512
        q = self.INPUT_SIZE // 4
        center = arr[q : 3 * q, q : 3 * q]
        for ch in range(3):
            hist_c, _ = np.histogram(center[:, :, ch], bins=32, range=(0, 256))
            feats.extend(hist_c.tolist())
        gray = np.dot(center[..., :3], [0.2989, 0.5870, 0.1140])
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        hist_g, _ = np.histogram(gx, bins=32, range=(0, 255))
        feats.extend(hist_g.tolist())
        vec = np.array(feats[: self.VECTOR_DIM], dtype=np.float32)
        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm
        return vec
