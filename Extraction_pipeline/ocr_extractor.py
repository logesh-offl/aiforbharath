"""EasyOCR-based text extraction for images and PDFs.

Images are OCR'd directly. PDFs are rasterized page-by-page to images
via PyMuPDF (no digital text-layer shortcut, per requirements: OCR is
applied uniformly to every page) and each page image is OCR'd the same
way as a standalone image.

The EasyOCR reader is expensive to construct (loads detection +
recognition models), so it is built once as a lazy module-level
singleton and reused across calls.

EasyOCR is used instead of PaddleOCR: paddlepaddle's macOS arm64 (Apple
Silicon) CPU wheel has a long-standing, widely-reported hang/deadlock
during inference (OpenBLAS/OpenMP threading issue), independent of this
codebase. EasyOCR (PyTorch-backed) runs natively on Apple Silicon
without that problem.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_OCR_ENGINE = None

# DPI used when rasterizing PDF pages to images before OCR.
PDF_RENDER_DPI = 200


def get_ocr_engine():
    """Lazily construct and cache the EasyOCR reader (singleton)."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        import easyocr

        _OCR_ENGINE = easyocr.Reader(["en"], gpu=False)
    return _OCR_ENGINE


def _load_image_array(image_path_or_bytes) -> np.ndarray:
    """Normalize an image source (path, bytes, file-like, PIL.Image, or
    ndarray) into an RGB numpy array PaddleOCR can consume."""
    if isinstance(image_path_or_bytes, np.ndarray):
        return image_path_or_bytes
    if isinstance(image_path_or_bytes, Image.Image):
        img = image_path_or_bytes
    elif isinstance(image_path_or_bytes, bytes):
        img = Image.open(io.BytesIO(image_path_or_bytes))
    elif hasattr(image_path_or_bytes, "read"):
        img = Image.open(image_path_or_bytes)
    else:
        img = Image.open(image_path_or_bytes)
    return np.array(img.convert("RGB"))


def _ocr_result_to_text(result) -> str:
    """Flatten EasyOCR's result list into plain text, one recognized
    line per output line, in reading order."""
    if not result:
        return ""
    lines = []
    for entry in result:
        # entry = (box, text, confidence)
        try:
            text = entry[1]
        except (IndexError, TypeError):
            continue
        if text:
            lines.append(text)
    return "\n".join(lines)


def extract_text_from_image(image_path_or_bytes) -> str:
    """Run EasyOCR on a single image and return the extracted text.

    Accepts a file path, raw bytes, a file-like object, a PIL.Image, or
    an already-decoded numpy array.
    """
    engine = get_ocr_engine()
    img_array = _load_image_array(image_path_or_bytes)
    result = engine.readtext(img_array)
    return _ocr_result_to_text(result)


def extract_text_from_pdf(pdf_path_or_bytes) -> str:
    """Rasterize every page of a PDF to an image (via PyMuPDF) and OCR
    each page with PaddleOCR, concatenating the results with page
    separators.

    Accepts a file path, raw bytes, or a file-like object.
    """
    import fitz  # PyMuPDF

    if isinstance(pdf_path_or_bytes, bytes):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    elif hasattr(pdf_path_or_bytes, "read"):
        doc = fitz.open(stream=pdf_path_or_bytes.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)

    zoom = PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    page_texts = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes(
                "RGB", (pix.width, pix.height), pix.samples)
            page_text = extract_text_from_image(image)
            page_texts.append(f"--- Page {page_index + 1} ---\n{page_text}")
    finally:
        doc.close()

    return "\n\n".join(page_texts)
