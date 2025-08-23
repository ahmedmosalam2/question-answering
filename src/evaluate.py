

from transformers import pipeline
import pandas as pd

def main():
    print("⚡ Using pretrained QA model...")

    # موديل جاهز متدرّب على SQuAD
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    # جربه على الداتا بتاعتك (مثلاً آخر 5 عينات بس عشان السرعة)
    df = pd.read_csv("data/train_processed.csv").tail(5)

    for idx, row in df.iterrows():
        question = row["question"]
        context = row["context"]
        true_answer = str(row["answer"])

        result = qa_pipeline(question=question, context=context)
        predicted = result["answer"]

        print(f"\nQ: {question}")
        print(f"True: {true_answer}")
        print(f"Pred: {predicted} (score={result['score']:.2f})")

if __name__ == "__main__":
    main()
