"""DocLens API — FastAPI bridge between the React frontend and the
local OCR + edge0 extraction pipeline (ocr_extractor.py, summarizer.py,
pipeline.py in the parent directory).

Both the EasyOCR engine and the edge0 model are loaded once at process
startup (module-level singletons) and reused across requests — edge0
in particular is expensive to build and serves one generation at a
time by design, so this process is meant to run as a single long-lived
server, not be spawned per-request.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the parent directory (ocr_extractor.py, summarizer.py,
# pipeline.py) importable regardless of the working directory this is
# launched from.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_BACKEND_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from ocr_extractor import get_ocr_engine  # noqa: E402
from pipeline import (  # noqa: E402
    SUPPORTED_INPUT_TYPES,
    infer_input_type,
    run_batch_pipeline,
    run_extraction_pipeline,
)
from summarizer import DEFAULT_MODEL_ENV_VAR, get_engine  # noqa: E402

# Fallback used when EDGE0_35B_MODEL isn't exported in the shell that
# launched the server: the known local checkout of the downloaded
# checkpoint, relative to this file's grandparent directory (repo root).
_LOCAL_FALLBACK_MODEL_DIR = os.path.join(
    os.path.dirname(_PIPELINE_DIR), "Edge0-35b-a3b-preview"
)


def _resolve_default_model_dir() -> str:
    env_dir = os.environ.get(DEFAULT_MODEL_ENV_VAR, "")
    if env_dir:
        return env_dir
    if os.path.isfile(os.path.join(_LOCAL_FALLBACK_MODEL_DIR, "config.json")):
        return _LOCAL_FALLBACK_MODEL_DIR
    return ""


_state: dict = {"ocr_engine": None, "edge0_engine": None, "model_dir": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the OCR engine eagerly; the edge0 engine is loaded lazily on
    # first request (so the server can start even before a model
    # directory is confirmed available) and then cached.
    _state["ocr_engine"] = get_ocr_engine()
    yield


app = FastAPI(title="DocLens API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_edge0_engine(model_dir: str | None):
    resolved = model_dir or _resolve_default_model_dir()
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "No edge0 model directory configured. Set "
                f"{DEFAULT_MODEL_ENV_VAR} or pass model_dir explicitly."
            ),
        )
    if _state["edge0_engine"] is None or _state["model_dir"] != resolved:
        try:
            _state["edge0_engine"] = get_engine(resolved)
            _state["model_dir"] = resolved
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500, detail=f"Failed to load edge0 model: {exc}"
            ) from exc
    return _state["edge0_engine"]


class ConfigResponse(BaseModel):
    default_model_dir: str
    supported_input_types: list[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        default_model_dir=_resolve_default_model_dir(),
        supported_input_types=list(SUPPORTED_INPUT_TYPES),
    )


@app.post("/api/process")
async def process_file(
    file: UploadFile = File(...),
    input_type: str = Form(...),
    instruction: str = Form(""),
    max_tokens: int = Form(256),
    model_dir: str = Form(""),
):
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"input_type must be one of {SUPPORTED_INPUT_TYPES}",
        )

    engine = _get_edge0_engine(model_dir or None)
    file_bytes = await file.read()

    try:
        result = run_extraction_pipeline(
            file_bytes, input_type, engine, max_tokens=max_tokens,
            instruction=instruction or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"filename": file.filename, "summary": result["summary"]}


@app.post("/api/batch")
async def process_batch(
    files: list[UploadFile] = File(...),
    instruction: str = Form(""),
    max_tokens: int = Form(256),
    model_dir: str = Form(""),
):
    engine = _get_edge0_engine(model_dir or None)

    file_payloads = []
    for f in files:
        file_payloads.append((f.filename, await f.read()))

    results = run_batch_pipeline(
        file_payloads, engine, max_tokens=max_tokens,
        instruction=instruction or None,
    )
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
