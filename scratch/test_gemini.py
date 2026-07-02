import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"Using API Key starting with: {api_key[:10] if api_key else 'None'}...")

client = genai.Client(api_key=api_key)

print("\n--- Trying to list models ---")
try:
    for m in client.models.list():
        print(f"Model Name: {m.name}, Supported Actions: {m.supported_actions}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\n--- Trying to generate content with gemini-2.5-flash ---")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, say test.",
    )
    print("Success with gemini-2.5-flash:", response.text)
except Exception as e:
    print("Failed with gemini-2.5-flash:", e)

print("\n--- Trying to generate content with gemini-1.5-flash ---")
try:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Hello, say test.",
    )
    print("Success with gemini-1.5-flash:", response.text)
except Exception as e:
    print("Failed with gemini-1.5-flash:", e)
