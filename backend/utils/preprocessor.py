import cv2
import numpy as np


def preprocess(image_bytes):
    """
    Takes raw image bytes from Flask request and returns
    a preprocessed numpy array ready for the detector.

    Steps:
        1. Decode bytes to numpy array
        2. Convert to grayscale
        3. Denoise
        4. Adaptive threshold
        5. Morphological cleanup

    Args:
        image_bytes (bytes): raw image file bytes

    Returns:
        processed (numpy array): preprocessed grayscale image
        or None if image could not be read
    """

    # ── Step 1: Decode bytes to image ─────────────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print("[preprocessor] Failed to decode image")
        return None

    # ── Step 2: Resize if too large (keep it manageable) ──────────────────────
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ── Step 3: Convert to grayscale ──────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 4: Denoise ───────────────────────────────────────────────────────
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # ── Step 5: Adaptive threshold (handles uneven lighting) ──────────────────
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # ── Step 6: Morphological cleanup (remove small noise) ────────────────────
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned