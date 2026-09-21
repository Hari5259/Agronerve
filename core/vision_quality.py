import io
import logging
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
from config import settings

logger = logging.getLogger(__name__)


class ImageQualityChecker:
    """Evaluates crop leaf image quality for sharpness, lighting, resolution, and foliage content."""

    def __init__(
        self,
        min_dim: int = 64,
        blur_threshold: float = settings.IMAGE_BLUR_THRESHOLD,
        min_leaf_ratio: float = settings.IMAGE_MIN_LEAF_RATIO,
    ):
        self.min_dim = min_dim
        self.blur_threshold = blur_threshold
        self.min_leaf_ratio = min_leaf_ratio

    def _calculate_laplacian_variance(self, gray_np: np.ndarray) -> float:
        """Calculates sharpness via discrete Laplacian filter variance."""
        # 3x3 Laplacian kernel
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        if gray_np.shape[0] < 3 or gray_np.shape[1] < 3:
            return 0.0

        padded = np.pad(gray_np.astype(np.float32), 1, mode="edge")
        lap = (
            padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            - 4.0 * gray_np
        )
        return float(np.var(lap))

    def evaluate_quality(self, image_bytes: bytes) -> Dict[str, Any]:
        """Performs comprehensive quality assessment on raw image bytes.

        Returns a dictionary containing:
            - is_usable (bool)
            - reason (str, optional)
            - resolution (Tuple[int, int])
            - blur_score (float)
            - luminance (float)
            - foliage_ratio (float)
        """
        if not image_bytes:
            return {
                "is_usable": False,
                "reason": "Image bytes are empty.",
                "blur_score": 0.0,
                "luminance": 0.0,
                "foliage_ratio": 0.0,
                "resolution": (0, 0),
            }

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {
                "is_usable": False,
                "reason": f"Corrupted or invalid image file: {str(e)}",
                "blur_score": 0.0,
                "luminance": 0.0,
                "foliage_ratio": 0.0,
                "resolution": (0, 0),
            }

        width, height = pil_img.size

        # 1. Resolution Check
        if width < self.min_dim or height < self.min_dim:
            return {
                "is_usable": False,
                "reason": f"Image resolution ({width}x{height}) is too small (minimum {self.min_dim}x{self.min_dim} required). Please capture a clearer image.",
                "blur_score": 0.0,
                "luminance": 0.0,
                "foliage_ratio": 0.0,
                "resolution": (width, height),
            }

        # Convert to numpy arrays for fast vectorized metrics
        img_np = np.array(pil_img)
        r, g, b = (
            img_np[:, :, 0].astype(np.float32),
            img_np[:, :, 1].astype(np.float32),
            img_np[:, :, 2].astype(np.float32),
        )

        # 2. Luminance & Exposure Check
        # Standard ITU-R BT.601 luma formula
        luminance_map = 0.299 * r + 0.587 * g + 0.114 * b
        mean_luminance = float(np.mean(luminance_map))

        if mean_luminance < 20.0:
            return {
                "is_usable": False,
                "reason": "Image is extremely dark / underexposed. Please capture a clearer image under adequate light.",
                "blur_score": 0.0,
                "luminance": round(mean_luminance, 2),
                "foliage_ratio": 0.0,
                "resolution": (width, height),
            }

        if mean_luminance > 240.0:
            return {
                "is_usable": False,
                "reason": "Image is extremely bright / washed out with glare. Please capture a clearer image.",
                "blur_score": 0.0,
                "luminance": round(mean_luminance, 2),
                "foliage_ratio": 0.0,
                "resolution": (width, height),
            }

        # 3. Foliage / Leaf Presence Ratio Check
        # Detect green foliage (G > R * 1.05 and G > B * 1.05) or necrotic/chlorotic leaf tissue (R > 70, G > 50, B < 140, R > B * 1.1)
        green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 30)
        brown_yellow_mask = (r > 70) & (g > 50) & (b < 140) & (r > b * 1.1)
        leaf_mask = green_mask | brown_yellow_mask
        foliage_ratio = float(np.sum(leaf_mask) / (width * height))

        if foliage_ratio < self.min_leaf_ratio:
            return {
                "is_usable": False,
                "reason": f"No crop leaf or foliage detected in image (foliage coverage {foliage_ratio*100:.1f}% < {self.min_leaf_ratio*100:.1f}%). Please position the camera over the crop leaf.",
                "blur_score": 0.0,
                "luminance": round(mean_luminance, 2),
                "foliage_ratio": round(foliage_ratio, 3),
                "resolution": (width, height),
            }

        # 4. Blur Detection (Laplacian Variance)
        # Downsample large images for consistent blur variance calculation
        scale_factor = min(1.0, 500.0 / max(width, height))
        if scale_factor < 1.0:
            eval_img = pil_img.resize(
                (int(width * scale_factor), int(height * scale_factor)),
                Image.Resampling.BILINEAR,
            )
            eval_gray = (
                np.array(eval_img.convert("L")).astype(np.float32)
            )
        else:
            eval_gray = luminance_map

        blur_score = self._calculate_laplacian_variance(eval_gray)
        if blur_score < self.blur_threshold:
            return {
                "is_usable": False,
                "reason": f"Image appears blurry (sharpness score {blur_score:.1f} < {self.blur_threshold}). Please hold camera steady and capture a clearer image.",
                "blur_score": round(blur_score, 2),
                "luminance": round(mean_luminance, 2),
                "foliage_ratio": round(foliage_ratio, 3),
                "resolution": (width, height),
            }

        return {
            "is_usable": True,
            "reason": None,
            "blur_score": round(blur_score, 2),
            "luminance": round(mean_luminance, 2),
            "foliage_ratio": round(foliage_ratio, 3),
            "resolution": (width, height),
        }


quality_checker = ImageQualityChecker()
