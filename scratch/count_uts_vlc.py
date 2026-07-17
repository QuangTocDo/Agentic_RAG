from datasets import load_dataset_builder

try:
    print("Connecting to Hugging Face and loading dataset metadata...")
    builder = load_dataset_builder("undertheseanlp/UTS_VLC")
    
    print("\n--- Document count per split in undertheseanlp/UTS_VLC ---")
    splits = builder.info.splits
    for split_name, split_info in splits.items():
        print(f"Split '{split_name}': {split_info.num_examples} documents")
except Exception as e:
    print(f"Error: {e}")
