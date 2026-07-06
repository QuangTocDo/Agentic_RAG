import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.environ.get("GOOGLE_API_KEY", "")

models_to_test = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-pro",
]

for model in models_to_test:
    print(f"\n--- Testing model: {model} ---")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=100
        )
        response = llm.invoke("Say hi")
        print(f"✅ Success! Response: {response.content.strip()}")
    except Exception as e:
        print(f"❌ Failed: {e}")
