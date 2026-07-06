from datasets import load_dataset

try:
    print("Loading corpus.parquet from YuITC/Vietnamese-Legal-Documents...")
    dataset = load_dataset("YuITC/Vietnamese-Legal-Documents", data_files="corpus.parquet")
    print("Dataset splits:")
    print(dataset)
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        print(dataset[split].features)
        print("First 3 items:")
        for i in range(3):
            print(f"\nItem {i}:")
            print(dataset[split][i])
except Exception as e:
    print(f"Error: {e}")
