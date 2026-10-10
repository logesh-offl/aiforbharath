# DocLens

Local OCR + summarization pipeline: upload an image or PDF, extract
its text with [EasyOCR](https://github.com/JaidedAI/EasyOCR), then run
it through the local [edge0](../edge0) model — either the default
summary, or a custom instruction of your choice. Exposed through a
React frontend backed by a small FastAPI service. Everything runs
locally; no data leaves the machine.

See [BLOG.md](BLOG.md) for the full writeup on the USP, the edge0
model architecture, and honest caveats.

## Architecture

```
frontend/  React + Vite + TypeScript — upload UI, single & batch modes
backend/   FastAPI — wraps ocr_extractor.py / summarizer.py / pipeline.py as HTTP endpoints
*.py       Core pipeline (OCR extraction, edge0 summarization, dispatch/batch logic)
```

The frontend never touches Python directly — it calls the FastAPI
backend over HTTP, which loads the EasyOCR and edge0 engines once and
keeps them resident (both are expensive to build and edge0 in
particular serves one generation at a time by design).

## Why EasyOCR, not PaddleOCR

PaddleOCR was the original choice, but `paddlepaddle`'s macOS arm64
(Apple Silicon) CPU wheel has a long-standing, widely reported
hang/deadlock during inference (an OpenBLAS/OpenMP threading issue,
independent of this codebase — confirmed while building this pipeline:
the process pinned to 100% CPU and never returned on a trivial image).
This is documented across multiple unrelated reports going back to
2022, and acknowledged by the PaddlePaddle maintainers as unresolved.
EasyOCR (PyTorch-backed) runs natively on Apple Silicon without this
problem, so it's used instead.

## Setup

### Python backend + pipeline

```bash
python3.12 -m venv .venv          # torch/easyocr wheels need <=3.13; 3.12 is safe
.venv/bin/pip install --upgrade pip setuptools
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../edge0             # installs the edge0 package this pipeline depends on
.venv/bin/pip install -r backend/requirements.txt
```

Point the pipeline at your downloaded edge0 checkpoint:

```bash
export EDGE0_35B_MODEL=/Users/logeshwarant/Downloads/Emos_chatbot/Edge0-35b-a3b-preview
```

If this isn't set, the backend falls back to that same path
automatically (see `_resolve_default_model_dir` in `backend/main.py`).
You can also override the model directory directly in the frontend's
Settings panel.

### React frontend

```bash
cd frontend
npm install
```

## Run

Two processes, both need to be running:

```bash
# Terminal 1 — backend (from Extraction_pipeline/)
cd backend
../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend (from Extraction_pipeline/)
cd frontend
npm run dev
```

Open the printed frontend URL (default `http://localhost:5173`). The
Vite dev server proxies `/api/*` and `/health` requests to the backend
on port 8000, so no CORS setup or manual base URL is needed.

Two modes are available via the toggle at the top:

- **Single file** — pick "Image" or "PDF", upload a file, optionally
  type a custom instruction, and click "Extract & Run". The first
  request loads both the OCR engine (at backend startup) and the
  edge0 model (lazily, on first request, then cached) — the edge0
  checkpoint load itself is only ~1s thanks to streaming SSD expert
  offload, but the very first request may take longer while model
  weights are read from disk.
- **Batch (multiple files)** — upload any mix of images and PDFs at
  once (file type is auto-detected per file from its extension) and
  click "Run Batch". edge0 serves one generation at a time by design,
  so files are processed sequentially on the backend, not in parallel.
  Results are shown per file (success/failure) and can be downloaded
  together as a single JSON file.

## Custom instructions

By default both modes summarize the extracted text. Typing something
into the "Instruction" field replaces that with whatever you ask for —
e.g. "extract only dates and amounts", "translate to English and
summarize", "list action items". In batch mode, the same instruction
is applied to every file.

## Project layout

- `ocr_extractor.py` — EasyOCR wrapper: `extract_text_from_image`,
  `extract_text_from_pdf` (PDF pages are rasterized to images via
  PyMuPDF at 200 DPI, then OCR'd the same way as a standalone image).
- `summarizer.py` — edge0 wrapper: `get_engine(model_dir)` builds the
  engine once; `run_instruction(text, engine, instruction, max_tokens)`
  runs a one-shot chat request, using the default summarization prompt
  when `instruction` is empty/None, or a custom-instruction prompt
  otherwise. `summarize(text, engine, max_tokens)` is a thin wrapper
  around `run_instruction` for the default-summary case.
- `pipeline.py` — `run_extraction_pipeline(file, input_type, engine, ...)`
  ties OCR + the edge0 instruction together for one file; returns
  `{"summary": ...}`. `run_batch_pipeline(files, engine, ...)` loops
  `run_extraction_pipeline` over a list of `(filename, file_bytes)`
  tuples, auto-detecting each file's type via `infer_input_type()` and
  isolating per-file failures so one bad file doesn't abort the batch.
- `backend/main.py` — FastAPI app. Loads the OCR engine at startup,
  loads the edge0 engine lazily on first request (cached after that).
  Endpoints: `GET /health`, `GET /api/config`, `POST /api/process`
  (single file), `POST /api/batch` (multiple files).
- `frontend/` — React + Vite + TypeScript UI: Settings sidebar (model
  directory, max tokens), a Single file / Batch mode toggle, file
  upload, optional instruction field, and result display (a single
  result card, or a per-file expandable list + JSON download for
  batches).
- `tests/` — Python unit tests. OCR tests are real (marked
  `@pytest.mark.slow`, ~20-30s, need model weights cached under
  `~/.EasyOCR`); summarizer and pipeline tests mock the edge0 engine
  so they don't need to load the real 19GB checkpoint.

## Testing

```bash
.venv/bin/python -m pytest                    # fast tests only (summarizer, pipeline)
.venv/bin/python -m pytest -m slow             # + real EasyOCR extraction tests
```

Frontend type-checking / build:

```bash
cd frontend
npx tsc --noEmit
npm run build
```

A manual smoke test against the real edge0 model (not part of the
automated suite, since loading a 19GB checkpoint per test run is
impractical) can be run with:

```bash
.venv/bin/python -c "
from summarizer import get_engine, summarize
engine = get_engine()  # reads EDGE0_35B_MODEL
print(summarize('Your text here...', engine))
"
```

Or exercise the full HTTP path once the backend is running:

```bash
curl -X POST http://127.0.0.1:8000/api/process \
  -F "file=@/path/to/sample.png" \
  -F "input_type=Image" \
  -F "max_tokens=80"
```

## Accuracy benchmark

`evaluation/` contains a small, reproducible benchmark that runs the
real pipeline (real EasyOCR, real edge0, no mocks) against four
synthetic documents with known ground truth, and reports OCR
Character/Word Error Rate plus field-extraction accuracy:

```bash
.venv/bin/pip install jiwer   # already in requirements.txt
.venv/bin/python evaluation/run_benchmark.py --json out.json
```

See [BLOG.md](BLOG.md)'s "Measured accuracy on this pipeline" section
for the current results and what they do/don't prove.

## Known limitations

- **No DOCX support.** EasyOCR operates on pixel images; DOCX is
  XML-based text with no native page-image representation, and
  converting it would require an extra system dependency (e.g.
  LibreOffice headless). Out of scope for now — image and PDF only.
- **No persistence.** Nothing is saved to disk server-side. Batch
  results can be downloaded as JSON from the browser, but there's no
  database or history.
- **No authentication.** The FastAPI backend has no auth layer. It's
  fine for local use; do not expose it on a network without adding
  access controls first.
- **Single active generation at a time.** Like edge0 itself, one
  generation runs at a time on the backend — concurrent requests queue
  behind each other. Batch mode makes this explicit: files are
  processed strictly one at a time, not in parallel.

See [BLOG.md](BLOG.md) for a fuller discussion of these tradeoffs,
including OCR accuracy caveats across languages and the measured
quality cost of 4-bit quantization.
