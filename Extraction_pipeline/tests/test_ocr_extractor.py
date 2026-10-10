"""Unit tests for ocr_extractor.

These tests build a small synthetic image / PDF with known text and
assert EasyOCR recovers that text. Real EasyOCR model weights are
downloaded on first use (cached under ~/.EasyOCR), so these tests are
slower than pure unit tests but exercise the real extraction path.
"""

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr_extractor import extract_text_from_image, extract_text_from_pdf


def _make_text_image(text: str, size=(600, 200)) -> Image.Image:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 60), text, fill="black", font=font)
    return img


@pytest.mark.slow
def test_extract_text_from_image_reads_known_text():
    img = _make_text_image("HELLO WORLD")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    text = extract_text_from_image(buf)

    assert "HELLO" in text.upper()
    assert "WORLD" in text.upper()


@pytest.mark.slow
def test_extract_text_from_pdf_reads_known_text_per_page():
    import fitz  # PyMuPDF

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 100), "FIRST PAGE TEXT", fontsize=24)
    page2 = doc.new_page()
    page2.insert_text((72, 100), "SECOND PAGE TEXT", fontsize=24)
    pdf_bytes = doc.tobytes()
    doc.close()

    text = extract_text_from_pdf(pdf_bytes)

    assert "FIRST PAGE" in text.upper()
    assert "SECOND PAGE" in text.upper()
    assert "Page 1" in text
    assert "Page 2" in text
