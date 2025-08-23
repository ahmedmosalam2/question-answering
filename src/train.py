


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset

def main():
    print("🤖 Starting training...")

    # Load processed data
    df = pd.read_csv("data/train_processed.csv")
    print(f"Loaded {len(df)} samples")

    # خذ عينة للتجريب
    df = df.head(3000)
    print(f"Using {len(df)} samples for training")

    # Create dataset
    dataset = Dataset.from_pandas(df)

    # Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Preprocess function (convert char → token positions)
    def preprocess_batch(batch):
        inputs = tokenizer(
            batch["question"],
            batch["context"],
            truncation=True,
            padding="max_length",
            max_length=384,
            return_offsets_mapping=True
        )

        start_positions = []
        end_positions = []

        for i, offset in enumerate(inputs["offset_mapping"]):
            start_char = batch["answer_start"][i]
            end_char = batch["answer_end"][i]

            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - list(reversed(sequence_ids)).index(1)

            start_token = end_token = 0
            for idx, (start, end) in enumerate(offset):
                if sequence_ids[idx] != 1:  # فقط التوكِنز بتاعة ال context
                    continue
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx

            start_positions.append(start_token)
            end_positions.append(end_token)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs.pop("offset_mapping")
        return inputs

    # Apply preprocessing
    dataset = dataset.map(preprocess_batch, batched=True, remove_columns=dataset.column_names)

    # Split data
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/qa_model",
        evaluation_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        warmup_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Training model...")
    trainer.train()

    # Save model
    trainer.save_model("models/qa_model")
    tokenizer.save_pretrained("models/qa_model")
    print("✅ Training completed! Model saved to models/qa_model")

if __name__ == "__main__":
    main()
