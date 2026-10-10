"""Run the DocLens benchmark: real OCR + real edge0 on a fixed set of
synthetic documents with known ground truth, and report CER/WER for
OCR and field-level accuracy for instruction-based extraction.

This exercises the REAL pipeline end to end (no mocks) — the same
ocr_extractor.extract_text_from_image and summarizer.run_instruction
functions the app itself calls. It's slow (loads EasyOCR + edge0) and
is meant to be run manually, not as part of the automated pytest
suite.

Usage:
    .venv/bin/python evaluation/run_benchmark.py
    .venv/bin/python evaluation/run_benchmark.py --model-dir /path/to/edge0-35b
    .venv/bin/python evaluation/run_benchmark.py --json out.json

Requires jiwer (pip install jiwer) for CER/WER computation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jiwer

sys.path.insert(0, ".")  # allow running from the evaluation/ dir too

from evaluation.dataset import BENCHMARK_SAMPLES, BenchmarkSample
from ocr_extractor import extract_text_from_image
from summarizer import get_engine, run_instruction


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so CER/WER measure real content
    errors, not incidental case/spacing differences from rendering."""
    return " ".join(text.lower().split())


def score_ocr(sample: BenchmarkSample, extracted_text: str) -> dict:
    ref = _normalize(sample.ground_truth_text)
    hyp = _normalize(extracted_text)
    cer = jiwer.cer(ref, hyp)
    wer = jiwer.wer(ref, hyp)
    return {"cer": round(cer, 4), "wer": round(wer, 4)}


def score_field_extraction(sample: BenchmarkSample, model_output: str) -> dict:
    """A field counts as correctly extracted if its expected value
    (case-insensitive) appears verbatim in the model's output. This is
    a strict, literal substring check — it will under-count paraphrased
    correct answers, which is intentional: for an extraction task we
    want the exact value reproduced, not a rewording."""
    output_lower = model_output.lower()
    per_field = {}
    for field_name, expected_value in sample.fields.items():
        per_field[field_name] = expected_value.lower() in output_lower
    correct = sum(per_field.values())
    total = len(per_field)
    return {
        "per_field": per_field,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
    }


def run(model_dir: str | None, max_tokens: int) -> dict:
    print("Loading edge0 engine...", file=sys.stderr)
    t0 = time.perf_counter()
    engine = get_engine(model_dir)
    print(f"  engine ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    results = []
    for sample in BENCHMARK_SAMPLES:
        print(f"Running sample: {sample.name}", file=sys.stderr)
        image_bytes = sample.render_png_bytes()

        t0 = time.perf_counter()
        extracted_text = extract_text_from_image(image_bytes)
        ocr_elapsed = time.perf_counter() - t0
        ocr_scores = score_ocr(sample, extracted_text)

        t0 = time.perf_counter()
        model_output = run_instruction(
            extracted_text, engine, instruction=sample.instruction,
            max_tokens=max_tokens)
        llm_elapsed = time.perf_counter() - t0
        field_scores = score_field_extraction(sample, model_output)

        results.append({
            "name": sample.name,
            "ocr": {**ocr_scores, "elapsed_s": round(ocr_elapsed, 2)},
            "extraction": {**field_scores, "elapsed_s": round(llm_elapsed, 2)},
            "raw_ocr_text": extracted_text,
            "raw_model_output": model_output,
        })

    engine.close()

    avg_cer = sum(r["ocr"]["cer"] for r in results) / len(results)
    avg_wer = sum(r["ocr"]["wer"] for r in results) / len(results)
    total_correct = sum(r["extraction"]["correct"] for r in results)
    total_fields = sum(r["extraction"]["total"] for r in results)

    summary = {
        "num_samples": len(results),
        "avg_ocr_cer": round(avg_cer, 4),
        "avg_ocr_wer": round(avg_wer, 4),
        "field_extraction_accuracy": (
            round(total_correct / total_fields, 4) if total_fields else None
        ),
        "fields_correct": total_correct,
        "fields_total": total_fields,
    }

    return {"summary": summary, "samples": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=None,
                     help="edge0 checkpoint dir (defaults to EDGE0_35B_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--json", default=None,
                     help="optional path to write full results as JSON")
    args = ap.parse_args()

    output = run(args.model_dir, args.max_tokens)

    print("\n=== DocLens Benchmark Summary ===")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")

    print("\n=== Per-sample detail ===")
    for r in output["samples"]:
        print(f"\n[{r['name']}]")
        print(f"  OCR   CER={r['ocr']['cer']}  WER={r['ocr']['wer']}"
              f"  ({r['ocr']['elapsed_s']}s)")
        print(f"  Field extraction: {r['extraction']['correct']}/"
              f"{r['extraction']['total']} correct"
              f"  ({r['extraction']['elapsed_s']}s)")
        for field_name, is_correct in r["extraction"]["per_field"].items():
            mark = "OK" if is_correct else "MISS"
            print(f"    [{mark}] {field_name}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull results written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
