from datasets import load_dataset

try:
    print("Loading namphan1999/data-luat...")
    dataset = load_dataset("namphan1999/data-luat")
    print("Dataset splits:")
    print(dataset)
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        print(dataset[split].features)
        print("First 3 items:")
        limit = min(3, len(dataset[split]))
        for i in range(limit):
            print(f"\nItem {i}:")
            print(dataset[split][i])
except Exception as e:
    print(f"Error: {e}")
