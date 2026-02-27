import boto3
import json
import base64
from pathlib import Path
from PIL import Image
import io
import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID = "amazon.nova-pro-v1:0"      # Nova Pro - best vision model in Nova family
REGION   = "us-east-1"                    # Nova Pro only available in us-east-1

client = boto3.client("bedrock-runtime", region_name=REGION)

# ─── OPTIMIZED SYSTEM PROMPT ────────────────────────────────────────────────
GARBAGE_SYSTEM_PROMPT = """
You are an advanced garbage detection and classification AI designed for public spaces.
Given an image, thoroughly analyze it and provide a detailed report:

1. Identify and list ALL visible garbage/waste items in the image.
2. Count the QUANTITY of each type of garbage.
3. Classify each garbage type from the following list:
   - plastic_bottle
   - plastic_bag
   - food_waste
   - paper_cardboard
   - glass_bottle
   - metal_can
   - electronic_waste
   - medical_waste
   - mixed_garbage
   - construction_debris
   - organic_waste
   - hazardous_waste
   - other

Ensure the output is formatted in valid JSON and follows this exact structure (no extra text, no markdown):
{
  "total_garbage_count": <number>,
  "severity_level": "",
  "garbage_types": [
    {
      "type": "",
      "quantity": ,
      "confidence": <0.0 to 1.0>,
      "location_in_image": ""
    }
  ],
  "cleanliness_score": <0 to 10>,
  "summary": ""
}

Provide the following additional details for optimal output:
- Each `garbage_type` should include a confidence level indicating the model's certainty in the classification.
- Specify the `location_in_image` where each garbage type is predominantly found.
- The `severity_level` should reflect the overall amount and type of waste detected.
- The `cleanliness_score` should be a numerical rating of the image’s cleanliness.
- The `summary` should concisely describe the overall condition of the scene.

Make sure the output is precise, clear, and concise, adhering strictly to the JSON format provided.
"""

# ─── INFERENCE PIPELINE ──────────────────────────────────────────────────────
class GarbageDetectionPipeline:

    def load_image(self, image_source: str) -> tuple:
        """Load image from local path or URL, returns (image_bytes, mime_type, pil_image)"""
        if image_source.startswith("http://") or image_source.startswith("https://"):
            r = requests.get(image_source, timeout=10)
            image_bytes = r.content
            mime_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        else:
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            ext = Path(image_source).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".webp": "image/webp",
                ".gif": "image/gif",  ".avif": "image/jpeg"  # avif → re-encode as jpeg
            }
            mime_type = mime_map.get(ext, "image/jpeg")

        # Nova Pro requires jpeg/png/gif/webp — re-encode avif or unknown formats
        pil_image = Image.open(io.BytesIO(image_bytes))
        if Path(image_source).suffix.lower() in [".avif", ".tiff", ".bmp"]:
            print(f"[INFO] Re-encoding {Path(image_source).suffix} → JPEG for Bedrock compatibility")
            buffer = io.BytesIO()
            pil_image.convert("RGB").save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            mime_type = "image/jpeg"

        return image_bytes, mime_type, pil_image

    def analyze(self, image_source: str) -> dict:
        """Send image to Nova Pro via Bedrock Converse API and return structured result."""
        print(f"[INFO] Loading image: {image_source}")
        image_bytes, mime_type, pil_image = self.load_image(image_source)
        print(f"[INFO] Image size: {pil_image.size}, Format: {mime_type}")

        # Extract format string for Bedrock (it needs "jpeg" not "image/jpeg")
        fmt = mime_type.split("/")[-1]   # e.g. "jpeg", "png", "webp", "gif"

        print(f"[INFO] Sending to {MODEL_ID}...")

        # ── Bedrock Converse API payload ──────────────────────────────────────
        response = client.converse(
            modelId=MODEL_ID,
            system=[
                {"text": GARBAGE_SYSTEM_PROMPT}
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": fmt,
                                "source": {
                                    "bytes": image_bytes   # raw bytes, no base64 needed with converse
                                }
                            }
                        },
                        {
                            "text": "Analyze this image for garbage/waste detection. Return structured JSON only."
                        }
                    ]
                }
            ],
            inferenceConfig={
                "temperature": 0.1,
                "maxTokens": 1024
            }
        )

        # ── Parse response ────────────────────────────────────────────────────
        raw_text = response["output"]["message"]["content"][0]["text"]

        # Strip markdown code fences if Nova wraps JSON in ```json... ```
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)
        result["image_source"] = image_source
        result["model_used"]   = MODEL_ID

        # ── Token usage ───────────────────────────────────────────────────────
        usage = response.get("usage", {})
        result["tokens_used"] = {
            "input":  usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0)
        }

        return result

    def print_report(self, result: dict):
        """Print a clean readable report."""
        print("\n" + "=" * 55)
        print("        GARBAGE DETECTION REPORT")
        print("=" * 55)
        print(f"  Image       : {result.get('image_source', 'N/A')}")
        print(f"  Model       : {result.get('model_used', 'N/A')}")
        print(f"  Total Count : {result.get('total_garbage_count', 0)} items")
        print(f"  Severity    : {result.get('severity_level', 'N/A').upper()}")
        print(f"  Cleanliness : {result.get('cleanliness_score', 0)}/10")
        print(f"  Summary     : {result.get('summary', '')}")
        if "tokens_used" in result:
            t = result["tokens_used"]
            print(f"  Tokens      : {t['input']} in / {t['output']} out")
        print("-" * 55)
        print("  GARBAGE BREAKDOWN:")
        print("-" * 55)
        for item in result.get("garbage_types", []):
            bar = "█" * int(item["confidence"] * 10)
            print(f"  • {item['type']:<25} Qty: {item['quantity']:<4} "
                  f"Conf: {bar:<10} ({item['confidence']:.0%})")
            print(f"    Location: {item['location_in_image']}")
        print("=" * 55)

    def save_result(self, result: dict, output_path: str = "garbage_result.json"):
        """Save full JSON result to file."""
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved → {output_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = GarbageDetectionPipeline()

    # ── Local image ──
    IMAGE_PATH = "/Users/logeshwarant/Downloads/Emos_chatbot/gabv2.avif"

    # ── Or URL ──
    # IMAGE_PATH = "https://upload.wikimedia.org/wikipedia/commons/9/92/Littering_on_beach.jpg"

    result = pipeline.analyze(IMAGE_PATH)
    pipeline.print_report(result)
    pipeline.save_result(result)