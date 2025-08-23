# 🧠 Question Answering System

A complete Question Answering (QA) system powered by **BERT**.  
It covers **data preprocessing, model training, evaluation, and inference**.

---

## 📋 Features

- **Data Preprocessing**: Clean and prepare SQuAD dataset  
- **Training**: Fine-tune BERT for QA  
- **Evaluation**: Measure performance with **Exact Match** & **F1 Score**  
- **Inference**: Interactive or batch-based question answering  

---

## 🚀 Quick Start


├── data/
│   ├── SQuAD-v1.1.csv          # Original dataset
│   └── train_processed.csv     # Preprocessed dataset
├── models/
│   └── qa_model/              # Trained model
├── src/
│   ├── preprocessing.py       # Data preprocessing
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   └── inference.py          # Inference / QA interface
├── run_pipeline.py           # Main pipeline script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation

# Requirements
```bash
pip install -r requirements.txt
```
