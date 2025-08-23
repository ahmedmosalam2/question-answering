import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_model(model_path="models/qa_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict_answer(question, context, model, tokenizer):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1

    answer_ids = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    confidence = torch.max(torch.softmax(start_logits, dim=-1)) * torch.max(torch.softmax(end_logits, dim=-1))
    confidence = confidence.item()

    return answer, confidence

def main():
    model, tokenizer = load_model("models/qa_model")

    while True:
        context = input("📜 Enter context (or 'exit' to quit): ")
        if context.lower() == "exit":
            break

        question = input("❓ Enter your question: ")

        answer, confidence = predict_answer(question, context, model, tokenizer)

        print("\n💬 Answer:", answer)
        print(f"📊 Confidence: {confidence:.2f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
