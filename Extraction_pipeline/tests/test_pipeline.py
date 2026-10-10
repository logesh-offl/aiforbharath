"""Unit tests for pipeline dispatch logic.

OCR and the edge0 instruction call are mocked here — this only
verifies that run_extraction_pipeline() picks the right extraction
function per input_type and wires its output into run_instruction(),
including passing through a custom instruction when supplied.
"""

from unittest.mock import MagicMock, patch

import pytest

import pipeline


def test_extract_text_dispatches_to_image_extractor():
    with patch("pipeline.extract_text_from_image") as mock_image, \
         patch("pipeline.extract_text_from_pdf") as mock_pdf:
        mock_image.return_value = "image text"

        result = pipeline.extract_text("file.png", pipeline.INPUT_TYPE_IMAGE)

        assert result == "image text"
        mock_image.assert_called_once_with("file.png")
        mock_pdf.assert_not_called()


def test_extract_text_dispatches_to_pdf_extractor():
    with patch("pipeline.extract_text_from_image") as mock_image, \
         patch("pipeline.extract_text_from_pdf") as mock_pdf:
        mock_pdf.return_value = "pdf text"

        result = pipeline.extract_text("file.pdf", pipeline.INPUT_TYPE_PDF)

        assert result == "pdf text"
        mock_pdf.assert_called_once_with("file.pdf")
        mock_image.assert_not_called()


def test_extract_text_raises_for_unsupported_type():
    with pytest.raises(ValueError):
        pipeline.extract_text("file.docx", "DOCX")


def test_run_extraction_pipeline_wires_ocr_output_into_run_instruction():
    fake_engine = MagicMock()
    with patch("pipeline.extract_text_from_image") as mock_image, \
         patch("pipeline.run_instruction") as mock_run_instruction:
        mock_image.return_value = "extracted raw text"
        mock_run_instruction.return_value = "final summary"

        result = pipeline.run_extraction_pipeline(
            "file.png", pipeline.INPUT_TYPE_IMAGE, fake_engine)

        assert result == {"summary": "final summary"}
        mock_image.assert_called_once_with("file.png")
        mock_run_instruction.assert_called_once_with(
            "extracted raw text", fake_engine, instruction=None,
            max_tokens=256)


def test_run_extraction_pipeline_passes_through_custom_instruction():
    fake_engine = MagicMock()
    with patch("pipeline.extract_text_from_pdf") as mock_pdf, \
         patch("pipeline.run_instruction") as mock_run_instruction:
        mock_pdf.return_value = "extracted raw text"
        mock_run_instruction.return_value = "2026-01-01, $42.00"

        result = pipeline.run_extraction_pipeline(
            "file.pdf", pipeline.INPUT_TYPE_PDF, fake_engine,
            max_tokens=64, instruction="extract only dates and amounts")

        assert result == {"summary": "2026-01-01, $42.00"}
        mock_run_instruction.assert_called_once_with(
            "extracted raw text", fake_engine,
            instruction="extract only dates and amounts", max_tokens=64)


def test_infer_input_type_recognizes_images_and_pdfs():
    assert pipeline.infer_input_type("scan.png") == pipeline.INPUT_TYPE_IMAGE
    assert pipeline.infer_input_type("photo.JPEG") == pipeline.INPUT_TYPE_IMAGE
    assert pipeline.infer_input_type("doc.pdf") == pipeline.INPUT_TYPE_PDF
    assert pipeline.infer_input_type("report.PDF") == pipeline.INPUT_TYPE_PDF
    assert pipeline.infer_input_type("notes.docx") is None
    assert pipeline.infer_input_type("no_extension") is None


def test_run_batch_pipeline_processes_each_file_and_reports_status():
    fake_engine = MagicMock()
    files = [("a.png", b"fake-image-bytes"), ("b.pdf", b"fake-pdf-bytes")]

    with patch("pipeline.run_extraction_pipeline") as mock_run:
        mock_run.side_effect = [
            {"summary": "summary A"},
            {"summary": "summary B"},
        ]

        results = pipeline.run_batch_pipeline(files, fake_engine)

    assert results == [
        {"filename": "a.png", "status": "ok", "summary": "summary A"},
        {"filename": "b.pdf", "status": "ok", "summary": "summary B"},
    ]
    assert mock_run.call_count == 2
    mock_run.assert_any_call(
        b"fake-image-bytes", pipeline.INPUT_TYPE_IMAGE, fake_engine,
        max_tokens=256, instruction=None)
    mock_run.assert_any_call(
        b"fake-pdf-bytes", pipeline.INPUT_TYPE_PDF, fake_engine,
        max_tokens=256, instruction=None)


def test_run_batch_pipeline_isolates_failures_per_file():
    fake_engine = MagicMock()
    files = [("good.png", b"ok-bytes"), ("bad.png", b"bad-bytes")]

    with patch("pipeline.run_extraction_pipeline") as mock_run:
        mock_run.side_effect = [
            {"summary": "all good"},
            RuntimeError("OCR blew up"),
        ]

        results = pipeline.run_batch_pipeline(files, fake_engine)

    assert results[0] == {
        "filename": "good.png", "status": "ok", "summary": "all good"}
    assert results[1]["filename"] == "bad.png"
    assert results[1]["status"] == "error"
    assert "OCR blew up" in results[1]["error"]


def test_run_batch_pipeline_flags_unsupported_extensions_without_calling_engine():
    fake_engine = MagicMock()
    files = [("notes.docx", b"docx-bytes")]

    with patch("pipeline.run_extraction_pipeline") as mock_run:
        results = pipeline.run_batch_pipeline(files, fake_engine)

    mock_run.assert_not_called()
    assert results == [{
        "filename": "notes.docx",
        "status": "error",
        "error": "Unsupported file type (expected image or PDF).",
    }]


def test_run_batch_pipeline_invokes_progress_callback_per_file():
    fake_engine = MagicMock()
    files = [("a.png", b"1"), ("b.pdf", b"2")]
    progress_calls = []

    with patch("pipeline.run_extraction_pipeline") as mock_run:
        mock_run.return_value = {"summary": "s"}
        pipeline.run_batch_pipeline(
            files, fake_engine,
            on_progress=lambda i, total, name: progress_calls.append(
                (i, total, name)))

    assert progress_calls == [(0, 2, "a.png"), (1, 2, "b.pdf")]


def test_run_batch_pipeline_passes_through_instruction_and_max_tokens():
    fake_engine = MagicMock()
    files = [("a.png", b"1")]

    with patch("pipeline.run_extraction_pipeline") as mock_run:
        mock_run.return_value = {"summary": "s"}
        pipeline.run_batch_pipeline(
            files, fake_engine, max_tokens=64, instruction="list action items")

    mock_run.assert_called_once_with(
        b"1", pipeline.INPUT_TYPE_IMAGE, fake_engine,
        max_tokens=64, instruction="list action items")
