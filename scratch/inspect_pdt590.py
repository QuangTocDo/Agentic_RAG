import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from configs.setting import settings

token = settings.hf_token
if token == "your_huggingface_token_here" or not token:
    token = None

try:
    print("Loading pdt590/vietnamese-legal-documents metadata...")
    meta_ds = load_dataset("pdt590/vietnamese-legal-documents", "metadata", split="data", streaming=True, token=token)
    print("Fetching first 5 items from metadata...")
    items = list(meta_ds.take(5))
    for i, item in enumerate(items):
        print(f"\nItem {i}:")
        for k, v in item.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
