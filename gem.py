from google import genai
from google.genai import types
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = ""
MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
GARBAGE_SYSTEM_PROMPT = """
You are an expert garbage detection and classification AI.
When given an image of a public place, analyze it and:

1. Detect ALL visible garbage/waste items in the image
2. Count the QUANTITY of each type
3. Classify each garbage type from this list:
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

Return ONLY valid JSON in this exact format (no extra text):
{
  "total_garbage_count": <number>,
  "severity_level": "<low|medium|high|critical>",
  "garbage_types": [
    {
      "type": "<garbage_type>",
      "quantity": <number>,
      "confidence": <0.0 to 1.0>,
      "location_in_image": "<top-left|top-right|center|bottom-left|bottom-right|spread>"
    }
  ],
  "cleanliness_score": <0 to 10>,
  "summary": "<one sentence summary>"
}
"""


# ─── INFERENCE PIPELINE ───────────────────────────────────────────────────────
class GarbageDetectionPipeline:
    def __init__(self):
        self.client = client

    def load_image(self, image_source: str) -> tuple:
        """Load image from local path or URL, return (PIL.Image, bytes, mime_type)"""
        if image_source.startswith("http://") or image_source.startswith("https://"):
            import requests
            response = requests.get(image_source, timeout=10)
            image_bytes = response.content
            mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        else:
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            suffix = Path(image_source).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp"}
            mime_type = mime_map.get(suffix, "image/jpeg")

        pil_image = Image.open(io.BytesIO(image_bytes))
        return pil_image, image_bytes, mime_type

    def analyze(self, image_source: str) -> dict:
        """Main method: send image to Gemini and get structured garbage analysis."""
        print(f"[INFO] Loading image: {image_source}")
        pil_image, image_bytes, mime_type = self.load_image(image_source)
        print(f"[INFO] Image size: {pil_image.size}, Format: {mime_type}")

        # Send to Gemini
        print(f"[INFO] Sending to {MODEL} for analysis...")
        
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text="Analyze this image for garbage/waste detection.")
            ],
            config=types.GenerateContentConfig(
                system_instruction=GARBAGE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        # Parse JSON response
        raw_text = response.text
        result = json.loads(raw_text)
        result["image_source"] = image_source
        result["model_used"] = MODEL
        return result

    def print_report(self, result: dict):
        """Print a clean readable report from the analysis result."""
        print("\n" + "="*55)
        print("       GARBAGE DETECTION REPORT")
        print("="*55)
        print(f"  Image       : {result.get('image_source', 'N/A')}")
        print(f"  Model       : {result.get('model_used', 'N/A')}")
        print(f"  Total Count : {result.get('total_garbage_count', 0)} items")
        print(f"  Severity    : {result.get('severity_level', 'N/A').upper()}")
        print(f"  Cleanliness : {result.get('cleanliness_score', 0)}/10")
        print(f"  Summary     : {result.get('summary', '')}")
        print("-"*55)
        print("  GARBAGE BREAKDOWN:")
        print("-"*55)
        for item in result.get("garbage_types", []):
            confidence_bar = "█" * int(item['confidence'] * 10)
            print(f"  • {item['type']:<25} Qty: {item['quantity']:<4} "
                  f"Conf: {confidence_bar:<10} ({item['confidence']:.0%})")
            print(f"    Location: {item['location_in_image']}")
        print("="*55)

    def save_result(self, result: dict, output_path: str = "garbage_result.json"):
        """Save full JSON result to file."""
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Result saved to: {output_path}")


# ─── MAIN ENTRYPOINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = GarbageDetectionPipeline()

    # ── Test with a local image ──
    IMAGE_PATH = "/Users/logeshwarant/Downloads/Emos_chatbot/gabv2.avif"   # <-- Change to your image path

    # ── OR test with a public URL ──
    # IMAGE_PATH = "https://upload.wikimedia.org/wikipedia/commons/9/92/Littering_on_beach.jpg"

    result = pipeline.analyze(IMAGE_PATH)
    pipeline.print_report(result)
    pipeline.save_result(result)
