"""edge0-backed text summarization.

Loads the edge0 model in-process (via the Python API, no HTTP server)
and summarizes arbitrary extracted text through a one-shot chat
request. The engine is expensive to build (streaming MoE expert
offload setup), so callers should build it once (e.g. via
``st.cache_resource`` in the Streamlit app) and reuse it across calls.
"""

from __future__ import annotations

import os

DEFAULT_MODEL_ENV_VAR = "EDGE0_35B_MODEL"
DEFAULT_MAX_SUMMARY_TOKENS = 256

SUMMARY_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. Summarize the user's "
    "text clearly and concisely, preserving the key facts, names, "
    "numbers, and dates. Do not add information that isn't present in "
    "the source text."
)

DEFAULT_INSTRUCTION = "Summarize the following text."

# System prompt used when the user supplies a custom instruction
# (e.g. "extract only dates and amounts", "translate to English and
# summarize", "list action items") instead of the default summary.
CUSTOM_INSTRUCTION_SYSTEM_PROMPT = (
    "You are a precise document assistant. Follow the user's "
    "instruction exactly, applying it only to the text provided. Do "
    "not add information that isn't present in the source text."
)


def resolve_model_dir(model_dir: str | None = None) -> str:
    """Resolve the edge0 checkpoint directory from an explicit argument
    or the ``EDGE0_35B_MODEL`` environment variable."""
    if model_dir:
        return model_dir
    env_dir = os.environ.get(DEFAULT_MODEL_ENV_VAR)
    if env_dir:
        return env_dir
    raise ValueError(
        f"No model_dir provided and {DEFAULT_MODEL_ENV_VAR} is not set. "
        "Pass a checkpoint directory explicitly or export "
        f"{DEFAULT_MODEL_ENV_VAR}=/path/to/edge0-35b."
    )


def get_engine(model_dir: str | None = None):
    """Build an edge0 engine for the resolved model directory.

    This is a heavyweight operation (loads the base checkpoint, LoRA
    adapters, and prerouter heads). Callers should build one engine and
    reuse it, not call this per-request.
    """
    from edge0 import AutoEngine

    resolved_dir = resolve_model_dir(model_dir)
    return AutoEngine.from_pretrained(resolved_dir)


def summarize(
    text: str,
    engine,
    max_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> str:
    """Summarize ``text`` using an already-built edge0 ``engine``.

    Builds a chat request with a summarization system prompt and the
    extracted text as the user turn, runs one generation, and returns
    the decoded summary text.
    """
    return run_instruction(
        text, engine, instruction=None, max_tokens=max_tokens)


def run_instruction(
    text: str,
    engine,
    instruction: str | None = None,
    max_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> str:
    """Apply an instruction to ``text`` using an already-built edge0
    ``engine``.

    If ``instruction`` is empty/None, falls back to the default
    summarization behavior (same prompt as ``summarize``). Otherwise
    the instruction replaces "summarize" with whatever the caller
    wants applied to the extracted text — e.g. "extract only dates and
    amounts", "translate to English and summarize", "list action
    items".
    """
    from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

    if not text or not text.strip():
        return ""

    instruction = (instruction or "").strip()
    if not instruction:
        system_prompt = SUMMARY_SYSTEM_PROMPT
        user_prompt = f"Summarize the following text:\n\n{text}"
    else:
        system_prompt = CUSTOM_INSTRUCTION_SYSTEM_PROMPT
        user_prompt = f"Instruction: {instruction}\n\nText:\n{text}"

    req = ChatRequest(
        model=engine.name,
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        max_tokens=max_tokens,
    )
    tokens, _meta = ChatSession(engine, req).run()
    return engine._tok.decode(tokens)
