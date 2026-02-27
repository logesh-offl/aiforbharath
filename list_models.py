import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDHXkMMRKXCBfbph3m3ChXEyd4if3YGkNE"
genai.configure(api_key=GEMINI_API_KEY)

print("Available models that support generateContent:\n")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  • {model.name}")
        print(f"    Display name: {model.display_name}")
        print()
