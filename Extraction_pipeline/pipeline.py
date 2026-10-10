"""End-to-end extraction pipeline: OCR (image/PDF) -> edge0 summary.

Dispatches to the right OCR extraction function based on the selected
input type, then runs the extracted text through edge0 — either the
default summarization prompt, or a custom user-supplied instruction
(e.g. "extract only dates and amounts", "translate to English and
summarize", "list action items"). Raw OCR text is intentionally not
returned (UI only shows the final result, per requirements).
"""

from __future__ import annotations

from ocr_extractor import extract_text_from_image, extract_text_from_pdf
from summarizer import run_instruction

INPUT_TYPE_IMAGE = "Image"
INPUT_TYPE_PDF = "PDF"

SUPPORTED_INPUT_TYPES = (INPUT_TYPE_IMAGE, INPUT_TYPE_PDF)


def extract_text(file, input_type: str) -> str:
    """Dispatch OCR extraction based on ``input_type``."""
    if input_type == INPUT_TYPE_IMAGE:
        return extract_text_from_image(file)
    if input_type == INPUT_TYPE_PDF:
        return extract_text_from_pdf(file)
    raise ValueError(
        f"Unsupported input_type {input_type!r}; expected one of "
        f"{SUPPORTED_INPUT_TYPES}")


def run_extraction_pipeline(
    file,
    input_type: str,
    engine,
    max_tokens: int = 256,
    instruction: str | None = None,
) -> dict:
    """Run OCR extraction then an edge0 instruction for a single
    uploaded file.

    Args:
        file: path, bytes, or file-like object for the uploaded document.
        input_type: one of ``SUPPORTED_INPUT_TYPES`` ("Image" or "PDF").
        engine: a pre-built edge0 engine (see ``summarizer.get_engine``).
        max_tokens: max tokens for the generated output.
        instruction: optional custom instruction to apply to the
            extracted text instead of the default summary (e.g.
            "extract only dates and amounts", "list action items").
            Falls back to summarization when empty/None.

    Returns:
        dict with a single ``"summary"`` key (raw OCR text is not
        exposed, per the UI requirements).
    """
    raw_text = extract_text(file, input_type)
    summary = run_instruction(
        raw_text, engine, instruction=instruction, max_tokens=max_tokens)
    return {"summary": summary}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def infer_input_type(filename: str) -> str | None:
    """Guess the input type (Image/PDF) from a filename's extension.

    Returns None if the extension isn't recognized (caller should skip
    or flag the file rather than guess wrong).
    """
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return INPUT_TYPE_IMAGE
    if ext in PDF_EXTENSIONS:
        return INPUT_TYPE_PDF
    return None


def run_batch_pipeline(
    files,
    engine,
    max_tokens: int = 256,
    instruction: str | None = None,
    on_progress=None,
) -> list[dict]:
    """Run the extraction pipeline over a batch of files sequentially.

    edge0 serves one generation at a time (single-slot engine, by
    design — see edge0's QueueServer), so batch processing here means
    looping through files one at a time rather than running them
    concurrently. Each file is processed independently: a failure on
    one file is captured in its result entry rather than aborting the
    whole batch.

    Args:
        files: iterable of (filename, file_bytes) tuples.
        engine: a pre-built edge0 engine (see ``summarizer.get_engine``).
        max_tokens: max tokens for each generated output.
        instruction: optional custom instruction applied to every file
            in the batch (same instruction for all files).
        on_progress: optional callback ``(index, total, filename)``
            invoked before processing each file, e.g. to drive a UI
            progress bar.

    Returns:
        A list of per-file result dicts, one per input file, each with
        keys ``filename``, ``status`` ("ok" or "error"), and either
        ``summary`` (on success) or ``error`` (on failure).
    """
    files = list(files)
    results = []
    for index, (filename, file_bytes) in enumerate(files):
        if on_progress:
            on_progress(index, len(files), filename)

        input_type = infer_input_type(filename)
        if input_type is None:
            results.append({
                "filename": filename,
                "status": "error",
                "error": "Unsupported file type (expected image or PDF).",
            })
            continue

        try:
            outcome = run_extraction_pipeline(
                file_bytes, input_type, engine, max_tokens=max_tokens,
                instruction=instruction)
        except Exception as exc:  # noqa: BLE001 — isolate per-file failures
            results.append({
                "filename": filename,
                "status": "error",
                "error": str(exc),
            })
            continue

        results.append({
            "filename": filename,
            "status": "ok",
            "summary": outcome["summary"],
        })

    return results
