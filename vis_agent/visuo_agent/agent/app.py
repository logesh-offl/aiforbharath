import asyncio
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from strands import Agent


class OutputFormat(Enum):
    FLUTTER = "flutter"
    TYPESCRIPT = "typescript"
    HTML_CSS = "html_css"


@dataclass
class CaptureResult:
    url: str
    html: str
    visible_text: str
    css_text: str
    full_page_png_b64: str
    viewport: Dict[str, Any]


async def capture_page(url: str, *, width: int = 1440, height: int = 900, wait_ms: int = 1500) -> CaptureResult:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": width, "height": height})
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(wait_ms)

        png_bytes = await page.screenshot(full_page=True, type="png")
        png_b64 = base64.b64encode(png_bytes).decode("utf-8")
        html = await page.content()

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        visible_text = " ".join(soup.get_text(" ").split())

        css_parts: List[str] = []
        inline_styles = await page.eval_on_selector_all(
            "style", "els => els.map(e => e.textContent || \'\').filter(Boolean)"
        )
        css_parts.extend(inline_styles)

        hrefs = await page.eval_on_selector_all(
            "link[rel=stylesheet]", "els => els.map(e => e.href).filter(Boolean)"
        )
        for href in hrefs:
            try:
                resp = await page.request.get(href)
                if resp.ok:
                    css_parts.append(await resp.text())
            except Exception:
                pass

        await context.close()
        await browser.close()

    return CaptureResult(
        url=url,
        html=html,
        visible_text=visible_text[:12000],
        css_text="\n\n".join(css_parts),
        full_page_png_b64=png_b64,
        viewport={"width": width, "height": height},
    )


# Flutter prompts
PUBSPEC_PROMPT = """You are a Flutter engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "pubspec.yaml", "content": "..."}
The pubspec must include google_fonts ^6.1.0 and assets/reference.png."""

MAIN_DART_PROMPT = """You are a Flutter engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "lib/main.dart", "content": "..."}

Requirements:
- Material 3, dark theme, google_fonts Inter
- Sections: hero, workflow cards, stats grid, CTA, articles, footer
- Single file, all widgets inside main.dart
- Use StatelessWidget/StatefulWidget properly
- Keep the code COMPLETE — do not truncate or add TODO placeholders"""

# TypeScript prompts
PACKAGE_JSON_PROMPT = """You are a TypeScript/React engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "package.json", "content": "..."}
Include React, TypeScript, Vite, and Tailwind CSS dependencies."""

TSCONFIG_PROMPT = """You are a TypeScript engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "tsconfig.json", "content": "..."}
Standard React + TypeScript configuration."""

APP_TSX_PROMPT = """You are a TypeScript/React engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "src/App.tsx", "content": "..."}

Requirements:
- Modern React with TypeScript and Tailwind CSS
- Responsive design matching the reference screenshot
- Sections: hero, workflow cards, stats grid, CTA, articles, footer
- Use functional components with hooks
- Keep the code COMPLETE — do not truncate or add TODO placeholders"""

# HTML/CSS prompts
HTML_PROMPT = """You are a frontend engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "index.html", "content": "..."}

Requirements:
- Modern semantic HTML5
- Responsive design matching the reference screenshot
- Sections: hero, workflow cards, stats grid, CTA, articles, footer
- Link to styles.css
- Keep the code COMPLETE — do not truncate or add TODO placeholders"""

CSS_PROMPT = """You are a frontend engineer. Output ONLY a raw JSON object, no markdown, no explanation.
JSON: {"path": "styles.css", "content": "..."}

Requirements:
- Modern CSS with custom properties
- Responsive design with media queries
- Clean, maintainable styles
- Match the reference screenshot design
- Keep the code COMPLETE — do not truncate or add TODO placeholders"""


def build_agent(system_prompt: str) -> Agent:
    return Agent(
        system_prompt=system_prompt,
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        tools=[],
        callback_handler=None,
    )


def clamp(s: str, n: int) -> str:
    return s[:n] if s else s


def extract_text_from_strands_result(result) -> str:
    """Drill into Strands result dict to get the actual text string."""
    if isinstance(result, dict):
        for block in result.get("content", []):
            if isinstance(block, dict) and "text" in block:
                return block["text"]
        return json.dumps(result)
    if hasattr(result, "message"):
        return extract_text_from_strands_result(result.message)
    return str(result)


def parse_file_output(result) -> Dict[str, str]:
    """Parse a single-file JSON response from the model."""
    text = extract_text_from_strands_result(result).strip()

    fence = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if fence:
        json_str = fence.group(1).strip()
    else:
        brace = re.search(r'\{[\s\S]*\}', text)
        json_str = brace.group(0).strip() if brace else text

    return json.loads(json_str)


def write_file(out_dir: str, file_obj: Dict[str, str]) -> None:
    path = os.path.join(out_dir, file_obj["path"])
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        w.write(file_obj["content"])


async def generate_flutter_from_url(url: str, out_dir: str = "./out_flutter", output_format: OutputFormat = OutputFormat.FLUTTER) -> str:
    cap = await capture_page(url)

    page_context = json.dumps({
        "url": cap.url,
        "viewport": cap.viewport,
        "visible_text": clamp(cap.visible_text, 4_000),
        "html_snippet": clamp(cap.html, 20_000),
        "css_snippet": clamp(cap.css_text, 20_000),
    })

    os.makedirs(out_dir, exist_ok=True)

    if output_format == OutputFormat.FLUTTER:
        # Step 1: pubspec.yaml
        pubspec_result = build_agent(PUBSPEC_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(pubspec_result))

        # Step 2: lib/main.dart
        main_dart_result = build_agent(MAIN_DART_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(main_dart_result))

        # Write screenshot asset
        assets_dir = os.path.join(out_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        with open(os.path.join(assets_dir, "reference.png"), "wb") as w:
            w.write(base64.b64decode(cap.full_page_png_b64))

    elif output_format == OutputFormat.TYPESCRIPT:
        # Step 1: package.json
        package_result = build_agent(PACKAGE_JSON_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(package_result))

        # Step 2: tsconfig.json
        tsconfig_result = build_agent(TSCONFIG_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(tsconfig_result))

        # Step 3: src/App.tsx
        app_tsx_result = build_agent(APP_TSX_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(app_tsx_result))

        # Write screenshot asset
        public_dir = os.path.join(out_dir, "public")
        os.makedirs(public_dir, exist_ok=True)
        with open(os.path.join(public_dir, "reference.png"), "wb") as w:
            w.write(base64.b64decode(cap.full_page_png_b64))

    elif output_format == OutputFormat.HTML_CSS:
        # Step 1: index.html
        html_result = build_agent(HTML_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(html_result))

        # Step 2: styles.css
        css_result = build_agent(CSS_PROMPT)(page_context)
        write_file(out_dir, parse_file_output(css_result))

        # Write screenshot asset
        assets_dir = os.path.join(out_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        with open(os.path.join(assets_dir, "reference.png"), "wb") as w:
            w.write(base64.b64decode(cap.full_page_png_b64))

    return out_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Capture a web UI and generate code in various formats.")
    parser.add_argument("url", help="Target UI URL")
    parser.add_argument("--out", default="out_flutter", help="Output directory")
    parser.add_argument(
        "--format",
        choices=["flutter", "typescript", "html_css"],
        default="flutter",
        help="Output format: flutter, typescript, or html_css (default: flutter)"
    )
    args = parser.parse_args()

    output_format = OutputFormat(args.format)
    out = asyncio.run(generate_flutter_from_url(args.url, args.out, output_format))
    sys.stdout.write(f"Project generated at: {out} (format: {args.format})\n")


if __name__ == "__main__":
    main()