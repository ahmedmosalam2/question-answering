


import pandas as pd

def main():
    print("📊 Loading and preprocessing data...")

    # Load raw data
    df = pd.read_csv("data/SQuAD-v1.1.csv")
    print(f"Loaded {len(df)} samples")

    # Clean data
    df = df.dropna(subset=["context", "question", "answer", "answer_start"])
    df = df.drop_duplicates()

    # بعض الـ datasets مفيهاش answer_end، فنحسبه
    if "answer_end" not in df.columns:
        df["answer_end"] = df.apply(
            lambda x: x["answer_start"] + len(str(x["answer"])), axis=1
        )

    # Save processed data
    df.to_csv("data/train_processed.csv", index=False)
    print(f"✅ Saved {len(df)} processed samples to data/train_processed.csv")

if __name__ == "__main__":
    main()
