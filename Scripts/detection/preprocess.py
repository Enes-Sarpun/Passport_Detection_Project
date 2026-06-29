from __future__ import annotations
import cv2
import numpy as np

# Each MRZ text line should be at least this tall after upscaling.
TARGET_LINE_HEIGHT = 80   # was 36 — too small for reliable OCR

def crop(image: np.ndarray, box: tuple[int, int, int, int], pad_frac: float = 0.03) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    pad_y = max(1, int((y2 - y1) * pad_frac))
    pad_x = max(1, int((x2 - x1) * pad_frac))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return image[y1:y2, x1:x2].copy()

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def deskew(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # np.where returns (rows, cols) = (y, x); transpose to (x, y) for minAreaRect.
    yx = np.column_stack(np.where(binary > 0))
    if len(yx) < 10:
        return gray
    xy = yx[:, ::-1].astype(np.float32)
    angle = cv2.minAreaRect(xy)[-1]
    # minAreaRect returns angle in [-90, 0); map to (-45, 45].
    if angle < -45:
        angle = 90 + angle
    # Only apply small corrections; large angles mean the detection box is wrong.
    if abs(angle) < 0.5 or abs(angle) > 10:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def upscale(gray: np.ndarray, n_lines: int = 2) -> np.ndarray:
    h = gray.shape[0]
    line_h = h / max(n_lines, 1)
    if line_h >= TARGET_LINE_HEIGHT * 0.8:
        return gray
    scale = TARGET_LINE_HEIGHT / line_h
    new_h = int(h * scale)
    new_w = int(gray.shape[1] * scale)
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clahe_aggressive(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _adaptive(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    bw = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _morpho_clean(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    _, bw = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Stretch histogram to full [0,255] range then apply CLAHE."""
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(norm)


def _suppress_guilloche(gray: np.ndarray, n_lines: int = 2) -> np.ndarray:
    h, w = gray.shape[:2]

    # Step 1: estimate background by blurring heavily.
    ksize = max(31, (w // 8) | 1)
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Step 2: subtract background (shift to 128 so we don't clip negatives).
    diff = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Step 3: large-block adaptive threshold per horizontal band.
    band_h = h // n_lines
    output = np.ones_like(diff) * 255  # white background
    for i in range(n_lines):
        y0 = i * band_h
        y1 = h if i == n_lines - 1 else (i + 1) * band_h
        band = diff[y0:y1, :]
        # Block size must be odd and cover at least one character width.
        block = max(31, (band.shape[1] // 20) | 1)
        bw = cv2.adaptiveThreshold(band, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block, 8)
        output[y0:y1, :] = bw

    # Light morphological opening to remove residual fine-pattern dots.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    return output


def _mark_filler_columns(binary: np.ndarray, n_lines: int) -> np.ndarray:
    marked = binary.copy()
    h, w = binary.shape

    # Estimate char width from image geometry (MRZ lines are 30/36/44 chars wide).
    line_lengths = (30, 36, 44)
    best_len = min(line_lengths, key=lambda l: abs(w / l - h / n_lines))
    char_w = w / best_len

    col_sums = binary.sum(axis=0)  # sum of pixel values per column

    # A column is "filler" if it has almost no ink (< 3% of max column sum).
    threshold = col_sums.max() * 0.03 if col_sums.max() > 0 else 1

    mid_y = h // 2
    mark_h = max(2, int(h * 0.08 / n_lines))

    # Only mark columns near expected filler positions (every char_w pixels).
    for char_idx in range(best_len):
        col_center = int((char_idx + 0.5) * char_w)
        # Check the column band around this expected center
        band_start = max(0, col_center - int(char_w * 0.3))
        band_end = min(w, col_center + int(char_w * 0.3))
        band_sum = col_sums[band_start:band_end].max() if band_end > band_start else 0
        if band_sum < threshold:
            # Draw a small mark so OCR registers this column as a character
            cv2.line(marked, (col_center, mid_y - mark_h),
                     (col_center, mid_y + mark_h), 0, 1)

    return marked

def preprocess(image: np.ndarray, box: tuple[int, int, int, int], n_lines: int = 2, *,
               do_deskew: bool = True, do_upscale: bool = True, do_clahe: bool = True,
               ) -> list[np.ndarray]:

    cropped = crop(image, box)
    gray = _to_gray(cropped)

    if do_deskew:
        gray = deskew(gray)

    if do_clahe:
        gray = _clahe(gray)

    # Sharpening before upscale keeps the kernel proportional to the font size.
    gray = _sharpen(gray)

    if do_upscale:
        gray = upscale(gray, n_lines=n_lines)

    otsu_img = _otsu(gray)
    adaptive_img = _adaptive(gray)
    raw_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu_binary)
    marked = _mark_filler_columns(inv, n_lines=n_lines)
    marked_bgr = cv2.cvtColor(cv2.bitwise_not(marked), cv2.COLOR_GRAY2BGR)

    norm_gray = _normalize_contrast(_to_gray(cropped))
    if do_deskew:
        norm_gray = deskew(norm_gray)
    if do_upscale:
        norm_gray = upscale(norm_gray, n_lines=n_lines)
    norm_img = _otsu(norm_gray)

    morpho_gray = _clahe_aggressive(_to_gray(cropped))
    if do_deskew:
        morpho_gray = deskew(morpho_gray)
    if do_upscale:
        morpho_gray = upscale(morpho_gray, n_lines=n_lines)
    morpho_img = _morpho_clean(morpho_gray)

    guilloche_base = _to_gray(cropped)
    if do_deskew:
        guilloche_base = deskew(guilloche_base)
    if do_upscale:
        guilloche_base = upscale(guilloche_base, n_lines=n_lines)
    guilloche_bw = _suppress_guilloche(guilloche_base, n_lines=n_lines)
    guilloche_img = cv2.cvtColor(guilloche_bw, cv2.COLOR_GRAY2BGR)

    return [otsu_img, adaptive_img, raw_bgr, marked_bgr, norm_img, morpho_img, guilloche_img]