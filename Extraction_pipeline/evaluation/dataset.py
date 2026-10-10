"""Small synthetic benchmark set for evaluating the DocLens pipeline.

This is NOT a claim of broad, statistically representative accuracy —
it's a fixed, reproducible set of documents with known ground truth,
used to get real numbers for *this* pipeline on *this* hardware,
instead of only citing published benchmarks for the underlying
libraries. See BLOG.md's Metrics section for how to interpret these
numbers and their limitations.

Each sample is a realistic single-page document type (invoice,
receipt, memo) rendered as an image, with:
  - ground_truth_text: the exact text placed on the image (for OCR
    CER/WER scoring against ocr_extractor's output).
  - fields: a dict of field -> expected value, used to score the
    custom-instruction extraction path (does edge0 correctly pull out
    specific fields from the OCR'd text?).
  - instruction: the custom instruction to send to edge0 for the
    field-extraction accuracy test.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _load_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _render_lines(lines: list[str], size=(900, 700), font_size=28) -> Image.Image:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += font_size + 18
    return img


@dataclass
class BenchmarkSample:
    name: str
    lines: list[str]
    fields: dict[str, str] = field(default_factory=dict)
    instruction: str = ""

    @property
    def ground_truth_text(self) -> str:
        return "\n".join(self.lines)

    def render_png_bytes(self) -> bytes:
        img = _render_lines(self.lines)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


BENCHMARK_SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample(
        name="invoice_01",
        lines=[
            "INVOICE",
            "Invoice Number: INV-58231",
            "Date: March 14 2026",
            "Customer: Riverside Hardware Co",
            "Description: 200 units of steel brackets",
            "Unit Price: 4.50 USD",
            "Total Due: 900.00 USD",
            "Payment Terms: Net 30 days",
        ],
        fields={
            "invoice_number": "INV-58231",
            "date": "March 14 2026",
            "total": "900.00 USD",
        },
        instruction=(
            "Extract only the invoice number, date, and total amount. "
            "Return them as a short list."
        ),
    ),
    BenchmarkSample(
        name="receipt_01",
        lines=[
            "GREENLEAF GROCERY",
            "Receipt No: 774920",
            "Date: 2026-02-02",
            "Item: Organic Rice 5kg  Price: 12.99",
            "Item: Olive Oil 1L      Price: 8.49",
            "Item: Whole Wheat Bread Price: 3.25",
            "Subtotal: 24.73",
            "Tax: 1.98",
            "Total: 26.71",
        ],
        fields={
            "receipt_number": "774920",
            "date": "2026-02-02",
            "total": "26.71",
        },
        instruction=(
            "Extract only the receipt number, date, and total amount. "
            "Return them as a short list."
        ),
    ),
    BenchmarkSample(
        name="memo_01",
        lines=[
            "INTERNAL MEMO",
            "From: Operations Team",
            "To: All Staff",
            "Date: April 9 2026",
            "Subject: Warehouse Safety Update",
            "Starting next Monday, all staff must wear",
            "high-visibility vests in the loading bay area.",
            "Hard hats are required near the east dock.",
            "Contact Priya Nair with any questions.",
        ],
        fields={
            "date": "April 9 2026",
            "subject": "Warehouse Safety Update",
            "contact": "Priya Nair",
        },
        instruction=(
            "Extract only the memo date, subject, and the contact "
            "person's name. Return them as a short list."
        ),
    ),
    BenchmarkSample(
        name="shipping_label_01",
        lines=[
            "SHIPPING LABEL",
            "Tracking Number: 1Z999AA10123456784",
            "Ship Date: 2026-05-20",
            "Recipient: Marcus Chen",
            "Address: 48 Oak Street, Springfield",
            "Weight: 3.4 kg",
            "Service: Express 2-Day",
        ],
        fields={
            "tracking_number": "1Z999AA10123456784",
            "recipient": "Marcus Chen",
            "service": "Express 2-Day",
        },
        instruction=(
            "Extract only the tracking number, recipient name, and "
            "shipping service. Return them as a short list."
        ),
    ),
]
