# UI → Flutter Automation Agent (Strands + Claude Haiku 4.5)

This project automates the manual workflow you described:

1. Open a URL in a browser.
2. Capture a **full-page screenshot** (equivalent to Chrome DevTools “Capture full size screenshot”).
3. Extract the page **HTML** and collect **CSS** (inline + external stylesheets).
4. Send these artifacts to an LLM agent (Strands Agents SDK) using **Claude Haiku 4.5**.
5. The agent generates a runnable **Flutter** project.
6. Optionally serve a local API endpoint (`localhost`) that triggers generation.

## Why Playwright (instead of DevTools)

Chrome DevTools UI actions (Command+Shift+P → “Capture full size screenshot”) are not easily automatable as a stable API.
Playwright provides a reliable programmatic equivalent: `page.screenshot(full_page=True)`.

## Prerequisites

- Python 3.10+
- Node.js (required by Playwright for browser binaries)
- Flutter SDK (to run the generated app)
- A Strands Agents configuration that can access Claude Haiku 4.5

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium
```

## Configure the model (Claude Haiku 4.5)

This repo sets the agent model as `claude-haiku-4-5` in `agent/app.py`.
How you authenticate depends on your Strands provider setup (Bedrock/Anthropic/etc.).
Refer to Strands Agents docs for provider configuration.

- Strands Agents SDK: https://strandsagents.com/latest/documentation/docs/  
- Strands SDK repo: https://github.com/strands-agents/sdk-python

## Run (CLI)

```bash
python -m agent.app "https://example.com"
```

Outputs a Flutter project in `./out_flutter/` with:

- `pubspec.yaml`
- `lib/main.dart`
- `README.md`
- `assets/reference.png`

Then run the generated Flutter app:

```bash
cd out_flutter
flutter pub get
flutter run
```

## Run (API server on localhost)

```bash
uvicorn server.api:app --reload --port 8000
```

Generate with:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

## Notes & limitations

- Some sites block automated browsers; you may need to set a custom user-agent, cookies, or run headed mode.
- CSS extraction is best-effort: it fetches inline `<style>` and external `link[rel=stylesheet]` content.
- Pixel-perfect Flutter recreation from HTML/CSS is not guaranteed; the agent aims for a close visual match.
